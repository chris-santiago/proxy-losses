"""
Per-class AP logging using return_per_class=True.

Demonstrates how to retrieve per-class loss tensors from SmoothAPLoss and
RecallAtQuantileLoss without a second forward pass. This pattern is useful
for monitoring which classes are improving and which are struggling during
training.

The return_per_class=True API returns:
    loss       : scalar (reduced loss as usual)
    per_class  : [C] tensor — per-class loss (NaN for degenerate classes)
    valid_mask : [C] bool tensor — True where class had both positives
                 and negatives in the pool

Always guard per-class logging with valid_mask to avoid logging NaN.

Usage:
    python examples/per_class_metrics_demo.py
    python examples/per_class_metrics_demo.py --loss recall
    python examples/per_class_metrics_demo.py --n-classes 4 --epochs 15
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from imbalanced_losses import SmoothAPLoss, RecallAtQuantileLoss, LossWarmupWrapper


# ── synthetic data ───────────────────────────────────────────────────────────


def make_data(
    n: int = 6_000,
    n_classes: int = 5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = [2.0 ** (-i) for i in range(n_classes)]
    total = sum(raw)
    weights = [r / total for r in raw]

    X, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=15,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=weights,
        flip_y=0.01,
        random_state=seed,
    )
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    split = int(0.8 * n)
    return X_t[:split], y_t[:split], X_t[split:], y_t[split:]


# ── model ────────────────────────────────────────────────────────────────────


class TinyMLP(nn.Module):
    def __init__(self, n_out: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── metric ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def macro_ap(model: nn.Module, X: torch.Tensor, y_np, n_classes: int) -> float:
    model.train(False)  # equivalent to model.eval()
    probs = torch.softmax(model(X), dim=1).cpu().numpy()
    y_bin = label_binarize(y_np, classes=range(n_classes))
    score = average_precision_score(y_bin, probs, average="macro")
    model.train()
    return float(score)


# ── training with per-class logging ─────────────────────────────────────────


def train_with_per_class_logging(
    n_classes: int = 5,
    epochs: int = 20,
    warmup_epochs: int = 5,
    blend_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    loss_type: str = "ap",
    seed: int = 0,
):
    X_train, y_train, X_val, y_val_np = make_data(n_classes=n_classes, seed=seed)

    torch.manual_seed(seed)
    model = TinyMLP(n_out=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_type == "ap":
        main_loss = SmoothAPLoss(num_classes=n_classes, queue_size=512)
        loss_name = "SmoothAPLoss"
    else:
        main_loss = RecallAtQuantileLoss(num_classes=n_classes, queue_size=512)
        loss_name = "RecallAtQuantileLoss"

    loss_fn = LossWarmupWrapper(
        warmup_loss=nn.CrossEntropyLoss(),
        main_loss=main_loss,
        warmup_epochs=warmup_epochs,
        blend_epochs=blend_epochs,
        temp_start=0.05,
        temp_end=0.005,
        temp_decay_steps=epochs * (len(X_train) // batch_size),
    )

    # Simple dict-based logger — accumulates per-class AP across batches each epoch
    per_class_log: dict[str, list[float]] = defaultdict(list)

    print(f"Training with {loss_name} (n_classes={n_classes})")
    print(f"{'Epoch':>5}  {'Phase':>6}  {'MacroAP':>8}  Per-class AP (valid classes)")
    print("-" * 70)

    global_step = 0
    for epoch in range(epochs):
        loss_fn.on_train_epoch_start(epoch)
        model.train()

        epoch_per_class: list[tuple[torch.Tensor, torch.Tensor]] = []

        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            loss_fn.on_train_batch_start(global_step)
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            if loss_fn.in_warmup or loss_fn.in_blend:
                # Warmup / blend — main loss not yet dominant; skip per-class log
                loss = loss_fn(model(xb), yb)
            else:
                # Main loss phase: forward for gradient update
                loss = loss_fn(model(xb), yb)

                # Separate no-grad pass to collect per-class breakdown for logging.
                # This avoids holding the computation graph for the logging call.
                with torch.no_grad():
                    logits = model(xb)
                _, per_class, valid = main_loss(logits, yb, return_per_class=True)
                epoch_per_class.append((per_class.detach(), valid))

            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1

        val_score = macro_ap(model, X_val, y_val_np, n_classes)
        phase = "warmup" if loss_fn.in_warmup else ("blend" if loss_fn.in_blend else "AP")

        # Aggregate per-class AP across batches (mean over valid batches only)
        if epoch_per_class:
            for c in range(n_classes):
                valid_vals = [
                    (1.0 - pc[c]).item()
                    for pc, vm in epoch_per_class
                    if vm[c].item()
                ]
                if valid_vals:
                    per_class_log[f"class_{c}"].append(sum(valid_vals) / len(valid_vals))

            class_str = "  ".join(
                f"c{c}={per_class_log[f'class_{c}'][-1]:.3f}"
                for c in range(n_classes)
                if per_class_log[f"class_{c}"]
            )
        else:
            class_str = "(warmup / blend)"

        print(f"{epoch:5d}  {phase:>6}  {val_score:8.4f}  {class_str}")

    print()
    print("Final per-class AP log (last 3 epochs, valid classes only):")
    for key, vals in sorted(per_class_log.items()):
        recent = vals[-3:]
        print(f"  {key}: {[f'{v:.4f}' for v in recent]}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--blend-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--loss-type",
        dest="loss_type",
        choices=["ap", "recall"],
        default="ap",
        help="main loss: 'ap' (SmoothAPLoss) or 'recall' (RecallAtQuantileLoss)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train_with_per_class_logging(**vars(args))
