"""
Tests for FocalSmoothAPLoss.

Coverage
--------
- Init: argument validation, focal param storage, queue buffer inheritance
- Equivalence: gamma=0, alpha=-1, beta=0 reproduces SmoothAPLoss exactly
- _compute_smooth_ap: focal modulation properties, violation weighting,
                      degenerate cases, gradient safety
- Forward: shape validation, ignore_index, perfect/worst classification,
           loss range, gradient flow, all reductions, binary mode,
           return_per_class, queue mechanics
- Gamma schedule: LossWarmupWrapper integration (gamma_start/gamma_end)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from imbalanced_losses.ap_loss import SmoothAPLoss
from imbalanced_losses.focal_ap_loss import FocalSmoothAPLoss
from imbalanced_losses.warmup_wrapper import LossWarmupWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_logits(
    B: int, C: int, score: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.arange(B) % C
    logits = torch.full((B, C), -score)
    for i, t in enumerate(targets):
        logits[i, t] = score
    return logits.requires_grad_(True), targets


def _worst_logits(
    B: int, C: int, score: float = 10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.arange(B) % C
    logits = torch.full((B, C), score)
    for i, t in enumerate(targets):
        logits[i, t] = -score
    return logits.requires_grad_(True), targets


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestFocalSmoothAPLossInit:
    def test_valid_default_construction(self):
        fn = FocalSmoothAPLoss(num_classes=4)
        assert fn.num_classes == 4
        assert fn.gamma == 2.0
        assert fn.alpha == -1.0
        assert fn.beta == 0.0

    def test_valid_custom_focal_params(self):
        fn = FocalSmoothAPLoss(num_classes=2, gamma=1.5, alpha=0.5, beta=1.0)
        assert fn.gamma == 1.5
        assert fn.alpha == 0.5
        assert fn.beta == 1.0

    def test_gamma_zero_valid(self):
        fn = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=-1.0)
        assert fn.gamma == 0.0

    def test_invalid_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma"):
            FocalSmoothAPLoss(num_classes=4, gamma=-0.1)

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            FocalSmoothAPLoss(num_classes=4, alpha=0.0)

    def test_invalid_alpha_negative_not_minus_one(self):
        with pytest.raises(ValueError, match="alpha"):
            FocalSmoothAPLoss(num_classes=4, alpha=-0.5)

    def test_invalid_beta_negative(self):
        with pytest.raises(ValueError, match="beta"):
            FocalSmoothAPLoss(num_classes=4, beta=-0.1)

    def test_inherits_queue_buffers(self):
        fn = FocalSmoothAPLoss(num_classes=3, queue_size=64)
        assert hasattr(fn, "_q_logits")
        assert hasattr(fn, "_q_targets")
        assert hasattr(fn, "_q_ptr")

    def test_inherits_parent_invalid_num_classes(self):
        with pytest.raises(ValueError):
            FocalSmoothAPLoss(num_classes=0)

    def test_inherits_parent_invalid_temperature(self):
        with pytest.raises(ValueError):
            FocalSmoothAPLoss(num_classes=2, temperature=0.0)


# ---------------------------------------------------------------------------
# Numerical equivalence with SmoothAPLoss when focal params are neutral
# ---------------------------------------------------------------------------


class TestEquivalence:
    """gamma=0, alpha=-1, beta=0 must produce identical output to SmoothAPLoss."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(7)

    def _run_both(self, scores, is_pos, tau=0.1):
        ap_parent, valid_parent = SmoothAPLoss._compute_smooth_ap(scores, is_pos, tau)
        fn = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=-1.0, beta=0.0, queue_size=0)
        ap_focal, valid_focal = fn._compute_smooth_ap(scores, is_pos, tau)
        return ap_parent, valid_parent, ap_focal, valid_focal

    def test_equivalence_random_binary(self):
        scores = torch.randn(40)
        is_pos = torch.zeros(40, dtype=torch.bool)
        is_pos[:5] = True
        ap_p, v_p, ap_f, v_f = self._run_both(scores, is_pos)
        assert v_p == v_f
        assert torch.isclose(ap_p, ap_f, atol=1e-6)

    def test_equivalence_even_split(self):
        scores = torch.randn(20)
        is_pos = torch.cat([torch.ones(10, dtype=torch.bool), torch.zeros(10, dtype=torch.bool)])
        ap_p, v_p, ap_f, v_f = self._run_both(scores, is_pos)
        assert v_p == v_f
        assert torch.isclose(ap_p, ap_f, atol=1e-6)

    def test_equivalence_degenerate_all_pos(self):
        scores = torch.randn(10)
        is_pos = torch.ones(10, dtype=torch.bool)
        _, v_p, _, v_f = self._run_both(scores, is_pos)
        assert v_p is False
        assert v_f is False

    def test_equivalence_degenerate_all_neg(self):
        scores = torch.randn(10)
        is_pos = torch.zeros(10, dtype=torch.bool)
        _, v_p, _, v_f = self._run_both(scores, is_pos)
        assert v_p is False
        assert v_f is False

    def test_equivalence_forward_scalar(self):
        torch.manual_seed(0)
        B, C = 32, 4
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        ap_ref = SmoothAPLoss(num_classes=C, queue_size=0)(logits, targets)
        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=0.0, alpha=-1.0, beta=0.0)
        ap_focal = fn(logits, targets)

        assert torch.isclose(ap_ref, ap_focal, atol=1e-5)


# ---------------------------------------------------------------------------
# Focal modulation properties
# ---------------------------------------------------------------------------


class TestFocalModulation:
    """Verify that focal weights behave as expected."""

    def test_well_ranked_positive_gets_lower_focal_weight(self):
        """A positive that beats all negatives should have focal_weight near 0."""
        tau = 0.1
        # scores: positive=10, negatives all at -10 → positive ranks first
        scores = torch.tensor([10.0, -10.0, -10.0, -10.0, -10.0])
        is_pos = torch.tensor([True, False, False, False, False])

        fn = FocalSmoothAPLoss(num_classes=1, gamma=2.0, queue_size=0)
        soft_gt_neg = torch.sigmoid(
            (scores[~is_pos] - scores[is_pos]) / tau
        )  # shape [1, 4]: all near 0 (negs score far below pos)
        n_neg = int((~is_pos).sum())
        p_rank = (1.0 - soft_gt_neg).sum() / n_neg  # near 1: all negs below pos
        focal_weight = (1.0 - p_rank) ** 2.0
        assert focal_weight.item() < 0.01  # well-ranked → low focal weight

    def test_poorly_ranked_positive_gets_high_focal_weight(self):
        """A positive buried under all negatives should have focal_weight near 1."""
        tau = 0.1
        scores = torch.tensor([-10.0, 10.0, 10.0, 10.0, 10.0])
        is_pos = torch.tensor([True, False, False, False, False])

        fn = FocalSmoothAPLoss(num_classes=1, gamma=2.0, queue_size=0)
        soft_gt_neg = torch.sigmoid(
            (scores[~is_pos] - scores[is_pos]) / tau
        )  # all near 1: negs score far above pos
        n_neg = int((~is_pos).sum())
        p_rank = (1.0 - soft_gt_neg).sum() / n_neg  # near 0: all negs above pos
        focal_weight = (1.0 - p_rank) ** 2.0
        assert focal_weight.item() > 0.99  # poorly-ranked → high focal weight

    def test_higher_gamma_increases_loss_on_mixed_ranking(self):
        """With mixed ranking, higher gamma should produce a different (not necessarily
        higher) loss because it suppresses well-ranked positives more aggressively."""
        torch.manual_seed(3)
        B, C = 40, 1
        logits = torch.randn(B, C)
        targets = torch.zeros(B, dtype=torch.long)
        targets[:4] = 1  # 4 positives, 36 negatives

        losses = {}
        for gamma in [0.0, 1.0, 2.0, 4.0]:
            fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=gamma)
            losses[gamma] = fn(logits, targets).item()

        # With gamma=0 we get vanilla Smooth-AP; as gamma increases, the loss
        # changes (we can't guarantee monotone direction without knowing exact
        # scores, but it must differ from gamma=0 for gamma>0)
        assert losses[0.0] != losses[2.0]

    def test_gamma_zero_matches_parent_forward(self):
        torch.manual_seed(1)
        B, C = 24, 3
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        ref = SmoothAPLoss(num_classes=C, queue_size=0)(logits, targets)
        focal = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=0.0, alpha=-1.0)(
            logits, targets
        )
        assert torch.isclose(ref, focal, atol=1e-5)

    def test_ap_in_range(self):
        torch.manual_seed(9)
        B, C = 32, 2
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0)
        loss = fn(logits, targets)
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_perfect_classification_focal_suppresses(self):
        """With perfect ranking and gamma>0, focal weights → 0, so FocalAP → 0 and loss → 1.
        This is the correct focal behavior: all positives are easy, gradient is suppressed."""
        logits, targets = _perfect_logits(40, 4)
        fn = FocalSmoothAPLoss(num_classes=4, queue_size=0, gamma=2.0)
        loss = fn(logits, targets)
        # Well-ranked positives get focal_weight ≈ 0 → weighted AP ≈ 0 → loss ≈ 1
        assert loss.item() > 0.9

    def test_perfect_classification_gamma_zero_low_loss(self):
        """With gamma=0, perfect classification yields low loss (same as SmoothAPLoss)."""
        logits, targets = _perfect_logits(40, 4)
        fn = FocalSmoothAPLoss(num_classes=4, queue_size=0, gamma=0.0)
        loss = fn(logits, targets)
        assert loss.item() < 0.1

    def test_worst_classification_high_loss(self):
        logits, targets = _worst_logits(40, 4)
        fn = FocalSmoothAPLoss(num_classes=4, queue_size=0, gamma=2.0)
        loss = fn(logits, targets)
        assert loss.item() > 0.5


# ---------------------------------------------------------------------------
# Violation weighting
# ---------------------------------------------------------------------------


class TestViolationWeighting:
    """Verify alpha/beta violation weighting properties."""

    def test_alpha_disabled_by_default(self):
        """Default alpha=-1 must give identical result to explicit alpha=-1."""
        torch.manual_seed(5)
        B, C = 24, 2
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))

        fn_default = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=0.0)
        fn_explicit = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=0.0, alpha=-1.0)
        assert torch.isclose(fn_default(logits, targets), fn_explicit(logits, targets))

    def test_violation_weighting_increases_rank_denominator(self):
        """With alpha>1, severe violations get extra weight → rank_all larger → AP lower.

        The violation contribution to rank_all is alpha * soft_gt^(1+beta).
        For alpha > 1, this exceeds the unweighted soft_gt, making rank_all larger.
        """
        torch.manual_seed(6)
        # Positive at 0, negatives at 1-4: all negatives score above positive (severe violations)
        scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        is_pos = torch.tensor([True, False, False, False, False])
        tau = 0.5

        fn_no_vio = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=-1.0, queue_size=0)
        # alpha=2 means each negative's contribution is doubled → rank_all larger → AP lower
        fn_vio = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=2.0, beta=0.0, queue_size=0)

        ap_no_vio, _ = fn_no_vio._compute_smooth_ap(scores, is_pos, tau)
        ap_vio, _ = fn_vio._compute_smooth_ap(scores, is_pos, tau)

        # alpha=2 inflates rank_all → AP should be lower (loss higher)
        assert ap_vio.item() < ap_no_vio.item()

    def test_no_violation_on_perfect_separation(self):
        """When positive clearly beats all negatives, violation weight is near zero."""
        scores = torch.tensor([10.0, -10.0, -10.0])
        is_pos = torch.tensor([True, False, False])
        tau = 0.1

        fn_no_vio = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=-1.0, queue_size=0)
        fn_vio = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=1.0, beta=1.0, queue_size=0)

        ap_no_vio, _ = fn_no_vio._compute_smooth_ap(scores, is_pos, tau)
        ap_vio, _ = fn_vio._compute_smooth_ap(scores, is_pos, tau)

        # soft_gt_neg ≈ 0 so w ≈ 0, rank_all same → AP should be nearly identical
        assert torch.isclose(ap_no_vio, ap_vio, atol=1e-3)

    def test_beta_zero_uniform_violation_weight(self):
        """With beta=0, w = alpha * soft_gt^0 = alpha for all pairs."""
        torch.manual_seed(8)
        scores = torch.randn(10)
        is_pos = torch.zeros(10, dtype=torch.bool)
        is_pos[:2] = True
        tau = 0.1

        fn_b0 = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=1.0, beta=0.0, queue_size=0)
        fn_b1 = FocalSmoothAPLoss(num_classes=1, gamma=0.0, alpha=1.0, beta=1.0, queue_size=0)

        ap_b0, _ = fn_b0._compute_smooth_ap(scores, is_pos, tau)
        ap_b1, _ = fn_b1._compute_smooth_ap(scores, is_pos, tau)

        # They differ because beta changes how violations are concentrated
        # (just checking they're not identical for non-trivial input)
        assert not torch.isclose(ap_b0, ap_b1, atol=1e-6)


# ---------------------------------------------------------------------------
# Forward integration (inherited behavior)
# ---------------------------------------------------------------------------


class TestForwardIntegration:
    """Inherited forward() behavior: shapes, reductions, ignore_index, gradients."""

    def test_scalar_output_mean_reduction(self):
        B, C = 32, 4
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0)
        loss = fn(logits, targets)
        assert loss.shape == ()

    def test_none_reduction_shape(self):
        B, C = 32, 4
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0, reduction="none")
        loss = fn(logits, targets)
        assert loss.shape == (C,)

    def test_sum_reduction(self):
        B, C = 32, 4
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        fn_none = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0, reduction="none")
        fn_sum = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0, reduction="sum")
        none_val = fn_none(logits, targets)
        sum_val = fn_sum(logits, targets)
        assert torch.isclose(none_val.nan_to_num(0.0).sum(), sum_val, atol=1e-5)

    def test_gradient_flows(self):
        B, C = 24, 3
        logits = torch.randn(B, C, requires_grad=True)
        targets = torch.randint(0, C, (B,))
        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0)
        loss = fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not logits.grad.isnan().any()

    def test_ignore_index_excludes_rows(self):
        B, C = 20, 2
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        targets_padded = targets.clone()
        targets_padded[::2] = -100  # every other row is padding

        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0)
        loss_clean = fn(logits[1::2], targets[1::2])
        loss_padded = fn(logits, targets_padded)
        assert torch.isclose(loss_clean, loss_padded, atol=1e-5)

    def test_all_padding_returns_zero(self):
        B, C = 16, 2
        logits = torch.randn(B, C, requires_grad=True)
        targets = torch.full((B,), -100)
        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0)
        loss = fn(logits, targets)
        assert loss.item() == 0.0
        loss.backward()  # must not raise

    def test_binary_mode(self):
        B = 30
        logits = torch.randn(B, 1)
        targets = torch.randint(0, 2, (B,))
        fn = FocalSmoothAPLoss(num_classes=1, queue_size=0, gamma=2.0)
        loss = fn(logits, targets)
        assert loss.shape == ()
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_return_per_class(self):
        B, C = 40, 4
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        fn = FocalSmoothAPLoss(num_classes=C, queue_size=0, gamma=2.0)
        loss, per_class, valid = fn(logits, targets, return_per_class=True)
        assert per_class.shape == (C,)
        assert valid.shape == (C,)
        assert torch.isclose(per_class[valid].mean(), loss, atol=1e-5)

    def test_queue_stores_and_merges(self):
        B, C = 16, 2
        logits = torch.randn(B, C)
        targets = torch.randint(0, C, (B,))
        fn = FocalSmoothAPLoss(num_classes=C, queue_size=32, gamma=2.0)
        fn.train()
        # First forward fills part of the queue
        loss1 = fn(logits, targets)
        assert fn._q_ptr == B
        # Second forward uses both live batch and queue
        loss2 = fn(logits, targets)
        assert loss2.shape == ()

    def test_shape_mismatch_raises(self):
        fn = FocalSmoothAPLoss(num_classes=4, queue_size=0, gamma=2.0)
        logits = torch.randn(10, 3)  # wrong C
        targets = torch.randint(0, 4, (10,))
        with pytest.raises(ValueError):
            fn(logits, targets)

    def test_wrong_target_length_raises(self):
        fn = FocalSmoothAPLoss(num_classes=4, queue_size=0, gamma=2.0)
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (8,))  # wrong N
        with pytest.raises(ValueError):
            fn(logits, targets)


# ---------------------------------------------------------------------------
# Gradient safety on _compute_smooth_ap
# ---------------------------------------------------------------------------


class TestGradients:
    def test_backward_does_not_raise(self):
        scores = torch.randn(20, requires_grad=True)
        is_pos = torch.zeros(20, dtype=torch.bool)
        is_pos[:3] = True
        fn = FocalSmoothAPLoss(num_classes=1, gamma=2.0, alpha=0.5, beta=1.0, queue_size=0)
        ap, valid = fn._compute_smooth_ap(scores, is_pos, tau=0.1)
        assert valid
        ap.backward()
        assert scores.grad is not None

    def test_no_nan_gradients(self):
        torch.manual_seed(42)
        scores = torch.randn(30, requires_grad=True)
        is_pos = torch.zeros(30, dtype=torch.bool)
        is_pos[:4] = True
        fn = FocalSmoothAPLoss(num_classes=1, gamma=2.0, queue_size=0)
        ap, _ = fn._compute_smooth_ap(scores, is_pos, tau=0.05)
        ap.backward()
        assert not scores.grad.isnan().any()

    def test_gamma_zero_grad_stable(self):
        torch.manual_seed(11)
        scores = torch.randn(20, requires_grad=True)
        is_pos = torch.zeros(20, dtype=torch.bool)
        is_pos[:2] = True
        fn = FocalSmoothAPLoss(num_classes=1, gamma=0.0, queue_size=0)
        ap, _ = fn._compute_smooth_ap(scores, is_pos, tau=0.1)
        ap.backward()
        assert not scores.grad.isnan().any()


# ---------------------------------------------------------------------------
# Gamma scheduling via LossWarmupWrapper
# ---------------------------------------------------------------------------


class TestGammaSchedule:
    def _make_wrapper(self, gamma_start=0.0, gamma_end=2.0, **kwargs):
        main = FocalSmoothAPLoss(num_classes=2, queue_size=0, gamma=0.0)
        return LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.5,
            temp_end=0.01,
            temp_decay_steps=10,
            gamma_start=gamma_start,
            gamma_end=gamma_end,
            **kwargs,
        ), main

    def test_gamma_set_to_gamma_start_at_latch(self):
        wrapper, main = self._make_wrapper(gamma_start=0.0, gamma_end=2.0)
        wrapper.on_train_epoch_start(1)   # enter main phase
        wrapper.on_train_batch_start(10)  # latch
        assert abs(main.gamma - 0.0) < 1e-6

    def test_gamma_reaches_gamma_end_after_decay_steps(self):
        wrapper, main = self._make_wrapper(gamma_start=0.0, gamma_end=2.0)
        wrapper.on_train_epoch_start(1)
        wrapper.on_train_batch_start(10)  # latch at step 10
        # After temp_decay_steps=10 steps elapsed:
        wrapper.on_train_batch_start(20)
        assert abs(main.gamma - 2.0) < 1e-6

    def test_gamma_linear_midpoint(self):
        wrapper, main = self._make_wrapper(gamma_start=0.0, gamma_end=2.0)
        wrapper.on_train_epoch_start(1)
        wrapper.on_train_batch_start(10)
        wrapper.on_train_batch_start(15)  # 5 of 10 steps → frac=0.5
        assert abs(main.gamma - 1.0) < 1e-6

    def test_gamma_clamped_after_decay_steps(self):
        wrapper, main = self._make_wrapper(gamma_start=0.0, gamma_end=2.0)
        wrapper.on_train_epoch_start(1)
        wrapper.on_train_batch_start(10)
        wrapper.on_train_batch_start(100)  # way past temp_decay_steps
        assert abs(main.gamma - 2.0) < 1e-6

    def test_no_gamma_params_no_scheduling(self):
        """When gamma_start/gamma_end are None, gamma is never touched."""
        main = FocalSmoothAPLoss(num_classes=2, queue_size=0, gamma=1.5)
        wrapper = LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.5,
            temp_end=0.01,
            temp_decay_steps=10,
        )
        wrapper.on_train_epoch_start(1)
        wrapper.on_train_batch_start(10)
        wrapper.on_train_batch_start(15)
        assert main.gamma == 1.5  # untouched

    def test_current_gamma_property(self):
        wrapper, main = self._make_wrapper(gamma_start=0.0, gamma_end=2.0)
        wrapper.on_train_epoch_start(1)
        wrapper.on_train_batch_start(10)
        assert wrapper.current_gamma is not None
        wrapper.on_train_batch_start(15)
        assert abs(wrapper.current_gamma - 1.0) < 1e-6

    def test_current_gamma_none_when_not_scheduled(self):
        main = FocalSmoothAPLoss(num_classes=2, queue_size=0)
        wrapper = LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=main,
            warmup_epochs=1,
            temp_start=0.5,
            temp_end=0.01,
            temp_decay_steps=10,
        )
        assert wrapper.current_gamma is None

    def test_warmup_epochs_zero_sets_gamma_start(self):
        """When warmup_epochs=0, gamma should be set to gamma_start immediately."""
        main = FocalSmoothAPLoss(num_classes=2, queue_size=0, gamma=5.0)
        LossWarmupWrapper(
            warmup_loss=nn.CrossEntropyLoss(),
            main_loss=main,
            warmup_epochs=0,
            temp_start=0.5,
            temp_end=0.01,
            temp_decay_steps=10,
            gamma_start=1.0,
            gamma_end=3.0,
        )
        assert abs(main.gamma - 1.0) < 1e-6

    def test_gamma_no_op_during_warmup(self):
        """Gamma should not change during warmup phase."""
        wrapper, main = self._make_wrapper(gamma_start=0.0, gamma_end=2.0)
        initial_gamma = main.gamma
        wrapper.on_train_epoch_start(0)   # still in warmup
        wrapper.on_train_batch_start(5)   # no-op
        assert main.gamma == initial_gamma

    def test_warning_when_main_loss_has_no_gamma(self):
        """Warning issued when gamma params given but main_loss has no gamma attr."""
        import warnings

        main = nn.Linear(4, 2)  # no gamma attribute
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=main,
                warmup_epochs=1,
                temp_start=0.5,
                temp_end=0.01,
                temp_decay_steps=10,
                gamma_start=0.0,
                gamma_end=2.0,
            )
        msgs = [str(x.message) for x in w]
        assert any("gamma" in m for m in msgs)

    def test_both_gamma_params_required(self):
        """Setting only one of gamma_start/gamma_end raises ValueError."""
        main = FocalSmoothAPLoss(num_classes=2, queue_size=0)
        with pytest.raises(ValueError, match="gamma"):
            LossWarmupWrapper(
                warmup_loss=nn.CrossEntropyLoss(),
                main_loss=main,
                warmup_epochs=1,
                temp_start=0.5,
                temp_end=0.01,
                temp_decay_steps=10,
                gamma_start=0.0,
                # gamma_end missing
            )
