"""
Side-by-side AUCPR comparison of four loss strategies on a class-imbalanced
binary classification task.

Strategies
----------
BCE           : vanilla BCEWithLogitsLoss — baseline, easy negatives dominate
SigmoidFocal  : focal loss (alpha=0.25, gamma=2) — down-weights easy examples
SmoothAP      : Smooth-AP loss — optimizes ranking directly via soft rank estimation
FocalSmoothAP : Focal-Smooth-AP with gamma schedule (0 → gamma_end) via
                LossWarmupWrapper.  Starts identical to SmoothAP (gamma=0) and
                ramps up focal difficulty weighting as easy positives become
                reliably well-ranked.  Using a fixed high gamma from epoch 0 is
                pathological: well-ranked positives immediately get focal_weight≈0,
                so the loss saturates at ≈1 regardless of model quality.

The key question: does a scheduled focal difficulty weighting *inside* Smooth-AP
improve over vanilla Smooth-AP on a severely imbalanced dataset?

Usage
-----
    python examples/ranking_demo.py
    python examples/ranking_demo.py --pos-rate 0.02     # less extreme imbalance
    python examples/ranking_demo.py --gamma 4.0         # stronger final focal effect
    python examples/ranking_demo.py --temperature 0.05  # sharper ranking
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score

from imbalanced_losses import (
    FocalSmoothAPLoss,
    LossWarmupWrapper,
    SigmoidFocalLoss,
    SmoothAPLoss,
)


# ── synthetic data ───────────────────────────────────────────────────────────


def make_data(
    n: int = 60_000,
    pos_rate: float = 0.005,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1 - pos_rate, pos_rate],
        flip_y=0.02,
        class_sep=0.8,
        random_state=seed,
    )
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    split = int(0.75 * n)
    return X_t[:split], y_t[:split], X_t[split:], y_t[split:]


# ── model ────────────────────────────────────────────────────────────────────


class TinyMLP(nn.Module):
    """Small MLP; outputs [N, 1] logits."""

    def __init__(self, d_in: int = 10, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [N, 1]


# ── loss adapters ────────────────────────────────────────────────────────────


class BCEAdapter(nn.Module):
    """BCEWithLogitsLoss; accepts [N, 1] logits and long targets."""

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits.view(-1), targets.float())


class FocalAdapter(nn.Module):
    """SigmoidFocalLoss; accepts [N, 1] logits and long targets."""

    def __init__(self, alpha: float, gamma: float):
        super().__init__()
        self.focal = SigmoidFocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits.view(-1), targets.float())


# ── evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    scores = model(X).view(-1).sigmoid().cpu().numpy()
    ap = average_precision_score(y.cpu().numpy(), scores)
    model.train()
    return float(ap)


# ── unified training loop ────────────────────────────────────────────────────


def run_one(
    loss_fn: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    total_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[float]:
    """Train a TinyMLP with loss_fn; return per-epoch val AUCPR.

    All four losses accept (logits [N, 1], targets [N] long) directly.
    SmoothAPLoss and FocalSmoothAPLoss use binary mode (num_classes=1)
    and build their ranking pool from the live batch plus their queue.
    If loss_fn is a LossWarmupWrapper, epoch/batch hooks are called so
    that temperature and gamma schedules advance correctly.
    """
    torch.manual_seed(seed)
    n = X_train.shape[0]
    model = TinyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    global_step = 0

    for epoch in range(total_epochs):
        if isinstance(loss_fn, LossWarmupWrapper):
            loss_fn.on_train_epoch_start(epoch)
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if isinstance(loss_fn, LossWarmupWrapper):
                loss_fn.on_train_batch_start(global_step)
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
        history.append(evaluate(model, X_val, y_val))

    return history


# ── compare ──────────────────────────────────────────────────────────────────


def compare(
    total_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    pos_rate: float = 0.005,
    queue_size: int = 4096,
    temperature: float = 0.01,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    gamma: float = 2.0,
    seed: int = 42,
):
    X_train, y_train, X_val, y_val = make_data(pos_rate=pos_rate, seed=seed)

    n_pos = int(y_train.sum().item())
    n_neg = len(y_train) - n_pos

    print(
        f"Dataset: n_train={len(X_train):,}  n_val={len(X_val):,}  "
        f"pos_rate={pos_rate}  n_pos={n_pos:,}  n_neg={n_neg:,}  "
        f"imbalance={n_neg/n_pos:.0f}:1\n"
    )

    n_batches = math.ceil(len(X_train) / batch_size)
    total_steps = total_epochs * n_batches

    shared = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        total_epochs=total_epochs, batch_size=batch_size,
        lr=lr, seed=seed,
    )

    focal_smooth_ap = LossWarmupWrapper(
        warmup_loss=BCEAdapter(),           # never used (warmup_epochs=0)
        main_loss=FocalSmoothAPLoss(
            num_classes=1,
            queue_size=queue_size,
            temperature=temperature,
            gamma=0.0,                      # overwritten by schedule; start at 0
        ),
        warmup_epochs=0,                    # no warmup — main loss active immediately
        temp_start=temperature,
        temp_end=temperature,               # constant temperature; only gamma changes
        temp_decay_steps=total_steps,
        gamma_start=0.0,
        gamma_end=gamma,
    )

    strategies = [
        (
            "BCE",
            BCEAdapter(),
        ),
        (
            "SigmoidFocal",
            FocalAdapter(alpha=focal_alpha, gamma=focal_gamma),
        ),
        (
            "SmoothAP",
            SmoothAPLoss(num_classes=1, queue_size=queue_size, temperature=temperature),
        ),
        (
            "FocalSmoothAP",
            focal_smooth_ap,
        ),
    ]

    print(
        f"BCE: vanilla  |  "
        f"SigmoidFocal: alpha={focal_alpha} gamma={focal_gamma}  |  "
        f"SmoothAP: τ={temperature} queue={queue_size}  |  "
        f"FocalSmoothAP: τ={temperature} queue={queue_size} γ 0→{gamma} over {total_steps} steps\n"
    )

    results: dict[str, list[float]] = {}
    for label, fn in strategies:
        print(f"Running {label}...")
        results[label] = run_one(fn, **shared)
        print(f"  best AUCPR: {max(results[label]):.4f}")

    # ── side-by-side table ───────────────────────────────────────────────────
    col = 15
    labels = list(results)
    header = f"\n{'epoch':>5}  " + "  ".join(f"{lbl:>{col}}" for lbl in labels)
    print(f"\nVal AUCPR per epoch")
    print(header)
    print("-" * len(header.lstrip("\n")))

    for epoch in range(total_epochs):
        row = f"{epoch:5d}  "
        row += "  ".join(f"{results[lbl][epoch]:{col}.4f}" for lbl in labels)
        print(row)

    print()
    best_row = "Best   " + "  ".join(f"{max(results[lbl]):{col}.4f}" for lbl in labels)
    print(best_row)
    print()

    winner = max(labels, key=lambda lbl: max(results[lbl]))
    print(f"Winner: {winner}  (best AUCPR = {max(results[winner]):.4f})")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--total-epochs", type=int, default=20,
                   help="training epochs per strategy (default: 20)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos-rate", type=float, default=0.005,
                   help="positive class rate (default: 0.005 → 200:1 imbalance)")
    p.add_argument("--queue-size", type=int, default=4096,
                   help="ranking memory queue capacity (default: 4096)")
    p.add_argument("--temperature", type=float, default=0.01,
                   help="sigmoid temperature for ranking losses (default: 0.01)")
    p.add_argument("--focal-alpha", type=float, default=0.25,
                   help="SigmoidFocalLoss alpha (default: 0.25)")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="SigmoidFocalLoss gamma (default: 2.0)")
    p.add_argument("--gamma", type=float, default=2.0,
                   help="FocalSmoothAPLoss final focal exponent; scheduled 0→gamma over training (default: 2.0)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    compare(**vars(args))
