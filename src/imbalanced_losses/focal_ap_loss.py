"""
FocalSmoothAPLoss — Focal-modulated Smooth-AP loss.

Extends SmoothAPLoss with two focal improvements applied inside the AP
computation, motivated by Lin et al. (2017):

1. **Rank-based focal modulation** — each positive's AP contribution is
   weighted by how difficult it is to rank:

       p_rank[k]      = (Σ_{j∈N} (1 − σ((s_j − s_k)/τ))) / |N|
                        # fraction of negatives ranked below positive k
       focal_weight[k] = (1 − p_rank[k])^γ
       FocalAP         = (1/|P|) Σ_k focal_weight[k] · rank_pos[k] / rank_all[k]

   When γ=0 this reduces exactly to standard Smooth-AP.

2. **Alpha-weighted pairwise violations** — severe violations (negatives
   clearly outscoring a positive) are up-weighted in the soft rank
   denominator:

       w[k,j] = α · σ((s_j − s_k)/τ)^β    for j ∈ N
       rank_all[k] = 1 + Σ_{j∈P} soft_gt[k,j] + Σ_{j∈N} w[k,j] · soft_gt[k,j]

   When α=−1 (default) this reduces exactly to the standard rank.

Both improvements vanish at their defaults (γ=0, α=−1, β=0), restoring
numerical equivalence with SmoothAPLoss.

Gamma can be scheduled jointly with temperature via LossWarmupWrapper's
gamma_start / gamma_end parameters.
"""

from __future__ import annotations

from typing import Literal

import torch

from imbalanced_losses.ap_loss import SmoothAPLoss


class FocalSmoothAPLoss(SmoothAPLoss):
    """
    Smooth-AP loss with focal-style difficulty weighting.

    Inherits all queue mechanics, DDP gathering, ignore_index handling,
    and reduction modes from :class:`SmoothAPLoss`.  Only the inner AP
    computation is overridden.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Use 1 for binary mode.
    queue_size : int, optional
        Memory queue capacity.  Default: 1024.
    temperature : float, optional
        Sigmoid sharpness τ.  Default: 0.01.
    reduction : {'mean', 'sum', 'none'}, optional
        Aggregation over classes.  Default: 'mean'.
    ignore_index : int, optional
        Target value to exclude.  Default: −100.
    update_queue_in_eval : bool, optional
        Update queue during eval.  Default: False.
    gather_distributed : bool or None, optional
        DDP all-gather control.  Default: None (auto-detect).
    gamma : float, optional
        Focal exponent γ ≥ 0.  Controls how strongly well-ranked
        positives are suppressed.  γ=0 disables focal modulation
        (exact equivalence with SmoothAPLoss).  Default: 2.0.
    alpha : float, optional
        Violation weight coefficient α.  Must be > 0 or exactly −1.
        −1 disables violation weighting (default).  When α > 0,
        pairwise (positive, negative) terms where the negative
        clearly outscores the positive receive extra weight.
    beta : float, optional
        Violation exponent β ≥ 0.  Controls the steepness of
        violation up-weighting.  Only used when α > 0.  β=0 applies
        uniform weight α to all negative pairs (constant up-weighting);
        β > 0 concentrates weight on the most severe violations.
        Default: 0.0.

    Notes
    -----
    Rank-based focal modulation is conceptually similar to score-based
    focal weighting, but operates in rank space rather than score space.
    Two positives with the same sigmoid confidence can have very different
    rank positions — focal modulation in rank space is more informative
    for retrieval objectives where AP depends on relative ordering.

    When α > 0 and β > 0, the violation weight w[k,j] is:

        w[k,j] = α · soft_gt[k,j]^β

    where soft_gt[k,j] ≈ P(s_j > s_k) is large when negative j clearly
    outscores positive k.  This up-weights the very pairs that most harm
    AP, directing gradient pressure toward resolving the worst violations
    first.

    Examples
    --------
    Basic usage (focal modulation only):

    >>> loss_fn = FocalSmoothAPLoss(num_classes=1, gamma=2.0)
    >>> loss = loss_fn(logits, targets)  # logits [N, 1], targets [N] in {0, 1}

    With violation weighting:

    >>> loss_fn = FocalSmoothAPLoss(num_classes=4, gamma=2.0, alpha=1.0, beta=1.0)

    Disable focal (recover SmoothAPLoss behaviour):

    >>> loss_fn = FocalSmoothAPLoss(num_classes=4, gamma=0.0, alpha=-1.0)

    Coupled temperature+gamma scheduling via LossWarmupWrapper:

    >>> from imbalanced_losses import LossWarmupWrapper
    >>> import torch.nn as nn
    >>> wrapper = LossWarmupWrapper(
    ...     warmup_loss=nn.CrossEntropyLoss(),
    ...     main_loss=FocalSmoothAPLoss(num_classes=4, queue_size=1024),
    ...     warmup_epochs=3,
    ...     temp_start=0.5,
    ...     temp_end=0.01,
    ...     temp_decay_steps=5000,
    ...     gamma_start=0.0,
    ...     gamma_end=2.0,
    ... )
    """

    def __init__(
        self,
        num_classes: int,
        queue_size: int = 1024,
        temperature: float = 0.01,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
        update_queue_in_eval: bool = False,
        gather_distributed: bool | None = None,
        *,
        gamma: float = 2.0,
        alpha: float = -1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            queue_size=queue_size,
            temperature=temperature,
            reduction=reduction,
            ignore_index=ignore_index,
            update_queue_in_eval=update_queue_in_eval,
            gather_distributed=gather_distributed,
        )

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha != -1.0 and alpha <= 0:
            raise ValueError(f"alpha must be > 0 or exactly -1 (disabled), got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _compute_smooth_ap(
        self,
        scores: torch.Tensor,
        is_pos: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, bool]:
        """
        Focal-modulated Smooth-AP for a single binary partition of the pool.

        Parameters
        ----------
        scores : torch.Tensor, shape [M]
            Raw scores for one class across the full pool (live + queue).
        is_pos : torch.Tensor, shape [M], dtype=bool
            True for positive samples.
        tau : float
            Sigmoid temperature.

        Returns
        -------
        ap : torch.Tensor, scalar
            Focal Smooth-AP estimate in [0, 1].
        valid : bool
            False for degenerate cases (all-positive or all-negative).

        Notes
        -----
        Steps 1-4 are identical to SmoothAPLoss._compute_smooth_ap.
        Steps 5-7 apply focal modulation and optional violation weighting.

        Pairwise soft indicator:
            diff[k,j]     = s_j − s_k         k ∈ P, j ∈ [M]
            soft_gt[k,j]  ≈ P(s_j > s_k) = σ(diff[k,j] / τ)

        Positive rank (unchanged from parent):
            rank_pos[k]   = 1 + Σ_{j∈P} soft_gt[k,j]   (self zeroed)

        Rank denominator with optional violation weighting (α > 0):
            w[k,j]        = α · soft_gt[k,j]^β          for j ∈ N
            rank_all[k]   = 1 + Σ_{j∈P} soft_gt[k,j]
                              + Σ_{j∈N} w[k,j] · soft_gt[k,j]

        Rank-based confidence and focal weight:
            p_rank[k]     = Σ_{j∈N} (1 − soft_gt[k,j]) / |N|
            focal[k]      = (1 − p_rank[k])^γ

        Focal-weighted AP:
            FocalAP = (1/|P|) · Σ_k focal[k] · rank_pos[k] / rank_all[k]

        When γ=0, α=−1, β=0: focal[k]=1 for all k, rank_all is standard,
        and FocalAP = mean(rank_pos / rank_all) — identical to parent.
        """
        m = scores.size(0)
        n_pos = int(is_pos.sum())

        if n_pos == 0 or n_pos == m:
            return scores.new_zeros(()), False

        # ── steps 1-4: pairwise diffs, soft indicators, self-mask ─────────
        pos_idx = is_pos.nonzero(as_tuple=False).squeeze(1)            # [P]
        diff_pos = scores.unsqueeze(0) - scores[pos_idx].unsqueeze(1)  # [P, M]
        soft_gt = torch.sigmoid(diff_pos / tau)                         # [P, M]

        self_mask = torch.zeros(n_pos, m, device=scores.device, dtype=torch.bool)
        self_mask[torch.arange(n_pos, device=scores.device), pos_idx] = True
        soft_gt = soft_gt.masked_fill(self_mask, 0.0)

        # ── step 5: split by class, compute ranks ──────────────────────────
        is_neg = ~is_pos
        n_neg = int(is_neg.sum())

        soft_gt_pos = soft_gt[:, is_pos]  # [P, |P|]  (includes self=0)
        soft_gt_neg = soft_gt[:, is_neg]  # [P, |N|]

        rank_pos = 1.0 + soft_gt_pos.sum(dim=1)  # [P]

        if self.alpha > 0 and n_neg > 0:
            # w[k,j] = alpha * soft_gt[k,j]^beta; concentrates on violations
            violation_w = self.alpha * soft_gt_neg.pow(self.beta)  # [P, |N|]
            neg_contrib = (violation_w * soft_gt_neg).sum(dim=1)   # [P]
        else:
            neg_contrib = soft_gt_neg.sum(dim=1)                   # [P]

        rank_all = 1.0 + soft_gt_pos.sum(dim=1) + neg_contrib  # [P]

        # ── step 6: rank-based focal weights ──────────────────────────────
        if self.gamma == 0.0:
            focal_weight = scores.new_ones(n_pos)
        else:
            # p_rank[k] = fraction of negatives ranked below positive k
            # (1 - soft_gt[k,j]) ≈ P(s_j < s_k); sum / |N| gives the fraction
            if n_neg > 0:
                p_rank = (1.0 - soft_gt_neg).sum(dim=1) / n_neg  # [P]
            else:
                p_rank = scores.new_ones(n_pos)
            focal_weight = (1.0 - p_rank).pow(self.gamma).detach()  # [P]

        # ── step 7: focal-weighted AP ──────────────────────────────────────
        ap = (focal_weight * rank_pos / rank_all).sum() / n_pos
        return ap, True


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    B, C = 32, 4
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))

    for gamma in [0.0, 1.0, 2.0]:
        loss_fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=gamma)
        loss = loss_fn(logits, targets)
        print(f"gamma={gamma:.1f}  loss={loss.item():.4f}")
    loss.backward()
    print("grad OK:", logits.grad is not None and not logits.grad.isnan().any())
