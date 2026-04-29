"""
Side-by-side macro-AP comparison: CE vs SoftmaxFocalLoss vs SmoothAPLoss+warmup.

Trains three models on the same imbalanced 5-class dataset and seed, then
prints per-epoch macro-average AP so you can see how each loss compares.

Strategies
----------
CE            : vanilla CrossEntropyLoss; majority class dominates gradients
focal         : SoftmaxFocalLoss with inverse-frequency alpha; down-weights
                easy majority-class examples and re-weights by class frequency
warmup+AP     : CrossEntropyLoss warmup then SmoothAPLoss; directly optimizes
                per-class AP and uses a memory queue for stable gradients at
                low positive rates

Usage:
    python examples/multiclass_demo.py
    python examples/multiclass_demo.py --n-classes 8
    python examples/multiclass_demo.py --total-epochs 30 --gamma 3.0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from imbalanced_losses import SmoothAPLoss, SoftmaxFocalLoss, LossWarmupWrapper


# ── synthetic data ───────────────────────────────────────────────────────────


def make_data(
    n: int = 10_000,
    n_classes: int = 5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Return train/val tensors and per-class counts for alpha computation."""
    weights = _imbalanced_weights(n_classes)
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
    split = int(0.75 * n)
    counts = [int((y_t[:split] == c).sum().item()) for c in range(n_classes)]
    return X_t[:split], y_t[:split], X_t[split:], y_t[split:], counts


def _imbalanced_weights(n_classes: int) -> list[float]:
    """Exponentially decreasing weights so tail classes are rare."""
    raw = [2.0 ** (-i) for i in range(n_classes)]
    total = sum(raw)
    return [r / total for r in raw]


# ── model ────────────────────────────────────────────────────────────────────


class TinyMLP(nn.Module):
    def __init__(self, d_in: int = 20, hidden: int = 128, n_out: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── metric ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def macro_ap(
    model: nn.Module,
    X: torch.Tensor,
    y_np,
    n_classes: int,
) -> float:
    model.eval()
    probs = torch.softmax(model(X), dim=1).cpu().numpy()
    y_bin = label_binarize(y_np, classes=range(n_classes))
    score = average_precision_score(y_bin, probs, average="macro")
    model.train()
    return float(score)


# ── training loops ───────────────────────────────────────────────────────────


def run_standard(
    loss_fn: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val,
    n_classes: int,
    total_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[float]:
    """Train with a standard (non-wrapper) loss; return per-epoch macro-AP."""
    torch.manual_seed(seed)
    n = X_train.shape[0]
    model = TinyMLP(n_out=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(total_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        history.append(macro_ap(model, X_val, y_val, n_classes))
    return history


def run_warmup(
    warmup_loss: nn.Module,
    main_loss: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val,
    n_classes: int,
    total_epochs: int,
    warmup_epochs: int,
    blend_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[float]:
    """Train with LossWarmupWrapper; return per-epoch macro-AP."""
    torch.manual_seed(seed)
    n = X_train.shape[0]
    model = TinyMLP(n_out=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = LossWarmupWrapper(
        warmup_loss=warmup_loss,
        main_loss=main_loss,
        warmup_epochs=warmup_epochs,
        blend_epochs=blend_epochs,
        temp_start=0.05,
        temp_end=0.005,
        temp_decay_steps=total_epochs * (n // batch_size),
    )

    history = []
    global_step = 0
    for epoch in range(total_epochs):
        loss_fn.on_train_epoch_start(epoch)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            loss_fn.on_train_batch_start(global_step)
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
        history.append(macro_ap(model, X_val, y_val, n_classes))
    return history


# ── compare ──────────────────────────────────────────────────────────────────


def compare(
    n_classes: int = 5,
    total_epochs: int = 20,
    warmup_epochs: int = 5,
    blend_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    gamma: float = 2.0,
    seed: int = 42,
):
    X_train, y_train, X_val, y_val_np, counts = make_data(
        n_classes=n_classes, seed=seed
    )

    # inverse-frequency alpha — down-weights majority class, up-weights rare ones
    total = sum(counts)
    alpha = [total / (n_classes * max(c, 1)) for c in counts]
    alpha_normed = [a / max(alpha) for a in alpha]  # normalize so max weight = 1

    print(f"Classes: {n_classes}  |  counts (train): {counts}")
    print(f"Focal alpha (normalized): {[f'{a:.3f}' for a in alpha_normed]}\n")

    shared = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val_np,
        n_classes=n_classes,
        total_epochs=total_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )

    print("Running CE (baseline)...")
    ce_hist = run_standard(nn.CrossEntropyLoss(), **shared)

    print("Running SoftmaxFocalLoss...")
    focal_hist = run_standard(
        SoftmaxFocalLoss(alpha=alpha_normed, gamma=gamma),
        **shared,
    )

    print("Running CE warmup → SmoothAPLoss...")
    ap_hist = run_warmup(
        warmup_loss=nn.CrossEntropyLoss(),
        main_loss=SmoothAPLoss(num_classes=n_classes, queue_size=1024),
        warmup_epochs=warmup_epochs,
        blend_epochs=blend_epochs,
        **shared,
    )

    # ── per-epoch table ──────────────────────────────────────────────────────
    labels = ["CE", f"Focal γ={gamma}", "Warmup+AP"]
    histories = [ce_hist, focal_hist, ap_hist]

    col = 12
    header = f"{'epoch':>5}  " + "  ".join(f"{lbl:>{col}}" for lbl in labels)
    print(f"\nVal macro-AP per epoch")
    print(header)
    print("-" * len(header))
    for epoch in range(total_epochs):
        row = f"{epoch:5d}  "
        row += "  ".join(f"{h[epoch]:{col}.4f}" for h in histories)
        print(row)

    bests = [max(h) for h in histories]
    print()
    print("Best val macro-AP  " + "  ".join(f"{lbl}: {b:.4f}" for lbl, b in zip(labels, bests)))


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--total-epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--blend-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    compare(**vars(args))
