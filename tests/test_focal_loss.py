"""
Tests for SigmoidFocalLoss and SoftmaxFocalLoss.

Sections
--------
1.  SigmoidFocalLoss – core correctness, alpha, reductions, gradients
2.  SoftmaxFocalLoss – core correctness (gamma=0 == CE)
3.  SoftmaxFocalLoss – alpha (per-class) weighting
4.  SoftmaxFocalLoss – reduction modes (none / mean / sum / mean_positive)
5.  SoftmaxFocalLoss – padding & ignore_index
6.  SoftmaxFocalLoss – mean_positive reduction
7.  SoftmaxFocalLoss – input shapes (2D, 3D, 4D)
8.  SoftmaxFocalLoss – gradient flow
9.  SoftmaxFocalLoss – numerical edge cases
10. SoftmaxFocalLoss – label smoothing
11. gather_distributed – DDP resolve / no-op at world_size=1
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from imbalanced_losses import SigmoidFocalLoss, SoftmaxFocalLoss

SEED = 42


# ===========================================================================
# Helpers
# ===========================================================================


def _init_single_process_group():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29501",
            world_size=1,
            rank=0,
        )


def _destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


# ===========================================================================
# 1. SigmoidFocalLoss
# ===========================================================================


class TestSigmoidFocalLoss:
    def test_gamma_zero_no_alpha_matches_bce(self):
        """gamma=0, alpha=-1 → must match F.binary_cross_entropy_with_logits."""
        torch.manual_seed(SEED)
        logits = torch.randn(32, 5)
        targets = torch.randint(0, 2, (32, 5)).float()

        focal = SigmoidFocalLoss(alpha=-1, gamma=0.0, reduction="mean")
        bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

        assert torch.allclose(focal(logits, targets), bce(logits, targets), atol=1e-6)

    def test_focal_downweights_easy_samples(self):
        """gamma>0 should produce element-wise loss <= gamma=0."""
        torch.manual_seed(SEED)
        logits = torch.randn(32, 5)
        targets = torch.randint(0, 2, (32, 5)).float()

        loss_g0 = SigmoidFocalLoss(alpha=-1, gamma=0.0, reduction="none")(logits, targets)
        loss_g2 = SigmoidFocalLoss(alpha=-1, gamma=2.0, reduction="none")(logits, targets)

        assert (loss_g2 <= loss_g0 + 1e-6).all()
        assert loss_g2.sum() > 0

    def test_alpha_scales_loss(self):
        """alpha in (0, 1) should apply positional weighting."""
        torch.manual_seed(SEED)
        logits = torch.randn(16, 4)
        targets = torch.ones(16, 4)  # all positives

        loss_no_alpha = SigmoidFocalLoss(alpha=-1, gamma=2.0, reduction="none")(logits, targets)
        loss_alpha    = SigmoidFocalLoss(alpha=0.75, gamma=2.0, reduction="none")(logits, targets)
        # All positives → alpha_t = alpha
        assert torch.allclose(loss_alpha, 0.75 * loss_no_alpha, atol=1e-6)

    def test_reductions(self):
        torch.manual_seed(SEED)
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 2, (16, 4)).float()

        loss_none = SigmoidFocalLoss(alpha=-1, gamma=2.0, reduction="none")(logits, targets)
        loss_sum  = SigmoidFocalLoss(alpha=-1, gamma=2.0, reduction="sum")(logits, targets)
        loss_mean = SigmoidFocalLoss(alpha=-1, gamma=2.0, reduction="mean")(logits, targets)

        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-6)
        assert torch.allclose(loss_mean, loss_none.mean(), atol=1e-6)

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            SigmoidFocalLoss(reduction="invalid")(torch.randn(4, 3), torch.zeros(4, 3))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            SigmoidFocalLoss(alpha=1.5)

    def test_gradient_flow(self):
        torch.manual_seed(SEED)
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 2, (16, 4)).float()
        loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert logits.grad.abs().sum() > 0

    def test_gather_distributed_attr(self):
        fn = SigmoidFocalLoss(gather_distributed=False)
        assert fn.gather_distributed is False
        assert fn._gather_resolved is None


# ===========================================================================
# 2. SoftmaxFocalLoss – core correctness
# ===========================================================================


class TestCoreFocalBehavior:
    def test_gamma_zero_matches_ce(self):
        """gamma=0, no alpha → must exactly match nn.CrossEntropyLoss."""
        torch.manual_seed(SEED)
        N, C = 64, 8
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        focal = SoftmaxFocalLoss(gamma=0.0, alpha=None, reduction="mean")
        ce = torch.nn.CrossEntropyLoss(reduction="mean")

        assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-6)

    def test_gamma_zero_matches_ce_with_padding(self):
        """gamma=0 with ignore_index must match nn.CrossEntropyLoss(ignore_index=...)."""
        torch.manual_seed(99)
        N, C = 32, 5
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        targets[24:] = -100

        focal = SoftmaxFocalLoss(gamma=0.0, reduction="mean", ignore_index=-100)
        ce = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

        assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-6)

    def test_focal_downweights_easy_samples(self):
        """gamma>0 should produce element-wise loss <= vanilla CE."""
        torch.manual_seed(SEED)
        N, C = 32, 6
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        loss_g0 = SoftmaxFocalLoss(gamma=0.0, reduction="none")(logits, targets)
        loss_g2 = SoftmaxFocalLoss(gamma=2.0, reduction="none")(logits, targets)

        assert (loss_g2 <= loss_g0 + 1e-6).all()
        assert loss_g2.sum() > 0

    def test_higher_gamma_stronger_focus(self):
        """Increasing gamma should more aggressively down-weight easy samples."""
        torch.manual_seed(SEED)
        N, C = 64, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        losses = {g: SoftmaxFocalLoss(gamma=g, reduction="mean")(logits, targets).item()
                  for g in [0.0, 0.5, 1.0, 2.0, 5.0]}

        for g_low, g_high in zip([0.0, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0, 5.0]):
            assert losses[g_low] >= losses[g_high] - 1e-6


# ===========================================================================
# 3. SoftmaxFocalLoss – alpha weighting
# ===========================================================================


class TestAlphaWeighting:
    def test_alpha_scales_per_class(self):
        torch.manual_seed(SEED)
        N, C = 128, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        alpha = [0.1, 0.2, 0.3, 0.4]

        loss_none  = SoftmaxFocalLoss(gamma=2.0, alpha=None,  reduction="none")(logits, targets)
        loss_alpha = SoftmaxFocalLoss(gamma=2.0, alpha=alpha, reduction="none")(logits, targets)

        ratios   = loss_alpha / (loss_none + 1e-12)
        expected = torch.tensor(alpha)[targets]
        assert torch.allclose(ratios, expected, atol=1e-5)

    def test_alpha_with_padding(self):
        torch.manual_seed(SEED)
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        targets[12:] = -100
        alpha = [0.1, 0.2, 0.3, 0.4]

        loss = SoftmaxFocalLoss(gamma=2.0, alpha=alpha, reduction="none", ignore_index=-100)(
            logits, targets
        )
        assert (loss[12:] == 0).all()
        assert (loss[:12] > 0).all()

    def test_uniform_alpha_equals_scaled(self):
        torch.manual_seed(SEED)
        N, C = 32, 5
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        k = 0.6

        loss_no  = SoftmaxFocalLoss(gamma=2.0, alpha=None,    reduction="none")(logits, targets)
        loss_uni = SoftmaxFocalLoss(gamma=2.0, alpha=[k] * C, reduction="none")(logits, targets)

        assert torch.allclose(loss_uni, k * loss_no, atol=1e-6)


# ===========================================================================
# 4. SoftmaxFocalLoss – reduction modes
# ===========================================================================


class TestReductions:
    def test_none_returns_full_shape(self):
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        loss = SoftmaxFocalLoss(gamma=2.0, reduction="none")(logits, targets)
        assert loss.shape == (N,)

    def test_sum_equals_none_sum(self):
        torch.manual_seed(SEED)
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        loss_none = SoftmaxFocalLoss(gamma=2.0, reduction="none")(logits, targets)
        loss_sum  = SoftmaxFocalLoss(gamma=2.0, reduction="sum")(logits, targets)
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-6)

    def test_mean_no_padding_equals_none_mean(self):
        torch.manual_seed(SEED)
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        loss_none = SoftmaxFocalLoss(gamma=2.0, reduction="none")(logits, targets)
        loss_mean = SoftmaxFocalLoss(gamma=2.0, reduction="mean")(logits, targets)
        assert torch.allclose(loss_mean, loss_none.mean(), atol=1e-6)

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            SoftmaxFocalLoss(gamma=2.0, reduction="invalid")(
                torch.randn(4, 3), torch.randint(0, 3, (4,))
            )


# ===========================================================================
# 5. SoftmaxFocalLoss – padding & ignore_index
# ===========================================================================


class TestPadding:
    def test_padded_positions_are_zero(self):
        torch.manual_seed(SEED)
        N, C = 16, 5
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        targets[0] = targets[7] = targets[15] = -100

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="none", ignore_index=-100)(logits, targets)
        assert loss[0].item() == 0.0
        assert loss[7].item() == 0.0
        assert loss[15].item() == 0.0
        assert (loss[[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]] > 0).all()

    def test_mean_divides_by_valid_count(self):
        """mean must divide by valid count, not total elements."""
        torch.manual_seed(SEED)
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        targets_padded = targets.clone()
        targets_padded[N // 2:] = -100

        fn = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)
        loss_padded = fn(logits, targets_padded)

        # Ground truth: mean only over the first-half valid positions
        loss_valid = SoftmaxFocalLoss(gamma=2.0, reduction="none")(
            logits[:N // 2], targets[:N // 2]
        ).mean()

        assert torch.allclose(loss_padded, loss_valid, atol=1e-6)

    def test_sum_unaffected_by_padding(self):
        torch.manual_seed(SEED)
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        n_valid = 10
        targets_padded = targets.clone()
        targets_padded[n_valid:] = -100

        loss_sum_padded = SoftmaxFocalLoss(gamma=2.0, reduction="sum", ignore_index=-100)(
            logits, targets_padded
        )
        loss_sum_sliced = SoftmaxFocalLoss(gamma=2.0, reduction="sum")(
            logits[:n_valid], targets[:n_valid]
        )
        assert torch.allclose(loss_sum_padded, loss_sum_sliced, atol=1e-5)

    def test_all_padded_returns_zero(self):
        N, C = 8, 4
        logits = torch.randn(N, C)
        targets = torch.full((N,), -100, dtype=torch.long)

        for reduction in ["none", "mean", "sum"]:
            loss = SoftmaxFocalLoss(gamma=2.0, reduction=reduction, ignore_index=-100)(
                logits, targets
            )
            if reduction == "none":
                assert (loss == 0).all()
            else:
                assert loss.item() == 0.0
            assert not torch.isnan(loss).any()
            assert not torch.isinf(loss).any()

    def test_single_valid_in_batch(self):
        torch.manual_seed(SEED)
        N, C = 8, 4
        logits = torch.randn(N, C)
        targets = torch.full((N,), -100, dtype=torch.long)
        targets[3] = 2

        loss_mean = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)(logits, targets)
        loss_none = SoftmaxFocalLoss(gamma=2.0, reduction="none", ignore_index=-100)(logits, targets)
        assert torch.allclose(loss_mean, loss_none[3], atol=1e-6)

    def test_custom_ignore_index(self):
        torch.manual_seed(SEED)
        N, C = 16, 5
        PAD = 0
        logits = torch.randn(N, C)
        targets = torch.randint(1, C, (N,))
        targets[12:] = PAD

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="none", ignore_index=PAD)(logits, targets)
        assert (loss[12:] == 0).all()
        assert (loss[:12] > 0).all()

        loss_mean = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=PAD)(logits, targets)
        assert torch.allclose(loss_mean, loss[:12].mean(), atol=1e-6)

    def test_negative_custom_ignore_index(self):
        torch.manual_seed(SEED)
        N, C = 16, 5
        PAD = -1
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))
        targets[10:] = PAD

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="none", ignore_index=PAD)(logits, targets)
        assert (loss[10:] == 0).all()
        assert (loss[:10] > 0).all()

        loss_mean = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=PAD)(logits, targets)
        assert torch.allclose(loss_mean, loss[:10].mean(), atol=1e-6)

    def test_varying_pad_ratios_stable_mean(self):
        torch.manual_seed(SEED)
        C = 4
        n_valid = 8
        logits_valid = torch.randn(n_valid, C)
        targets_valid = torch.randint(0, C, (n_valid,))

        reference = SoftmaxFocalLoss(gamma=2.0, reduction="mean")(
            logits_valid, targets_valid
        ).item()

        for n_pad in [0, 8, 32, 128]:
            N = n_valid + n_pad
            logits = torch.randn(N, C)
            logits[:n_valid] = logits_valid
            targets = torch.full((N,), -100, dtype=torch.long)
            targets[:n_valid] = targets_valid

            loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)(
                logits, targets
            ).item()
            assert abs(loss - reference) < 1e-5


# ===========================================================================
# 6. SoftmaxFocalLoss – mean_positive reduction
# ===========================================================================


class TestMeanPositive:
    def test_divides_by_positive_count(self):
        torch.manual_seed(SEED)
        N, C = 32, 5
        logits = torch.randn(N, C)
        targets = torch.zeros(N, dtype=torch.long)
        targets[:8] = torch.randint(1, C, (8,))

        loss_none = SoftmaxFocalLoss(gamma=2.0, reduction="none", background_class=0)(
            logits, targets
        )
        loss_mp = SoftmaxFocalLoss(gamma=2.0, reduction="mean_positive", background_class=0)(
            logits, targets
        )

        expected = loss_none.sum() / 8
        assert torch.allclose(loss_mp, expected, atol=1e-6)

        loss_mean = SoftmaxFocalLoss(gamma=2.0, reduction="mean", background_class=0)(
            logits, targets
        )
        assert loss_mp > loss_mean

    def test_negatives_contribute_to_numerator(self):
        torch.manual_seed(SEED)
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets_all_pos = torch.randint(1, C, (N,))
        targets_half = targets_all_pos.clone()
        targets_half[8:] = 0

        loss_none_half = SoftmaxFocalLoss(gamma=2.0, reduction="none", background_class=0)(
            logits, targets_half
        )
        # Background positions should still produce non-zero unreduced loss
        assert loss_none_half[8:].sum() > 0

        loss_half = SoftmaxFocalLoss(
            gamma=2.0, reduction="mean_positive", background_class=0
        )(logits, targets_half)
        assert loss_half.item() > 0

    def test_mean_positive_with_padding(self):
        torch.manual_seed(SEED)
        N, C = 32, 5
        logits = torch.randn(N, C)
        targets = torch.zeros(N, dtype=torch.long)
        targets[:8] = torch.randint(1, C, (8,))
        targets[24:] = -100  # 8 padded

        loss = SoftmaxFocalLoss(
            gamma=2.0, reduction="mean_positive", ignore_index=-100, background_class=0
        )(logits, targets)
        loss_none = SoftmaxFocalLoss(
            gamma=2.0, reduction="none", ignore_index=-100, background_class=0
        )(logits, targets)

        expected = loss_none.sum() / 8
        assert torch.allclose(loss, expected, atol=1e-6)
        assert (loss_none[24:] == 0).all()

    def test_no_positives_no_nan(self):
        N, C = 16, 4
        logits = torch.randn(N, C)
        targets = torch.zeros(N, dtype=torch.long)  # all background

        loss = SoftmaxFocalLoss(
            gamma=2.0, reduction="mean_positive", background_class=0
        )(logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_custom_background_class(self):
        torch.manual_seed(SEED)
        N, C = 32, 5
        BG = 3
        logits = torch.randn(N, C)
        targets = torch.full((N,), BG, dtype=torch.long)
        targets[:10] = torch.arange(10) % BG  # classes 0,1,2 — all non-BG

        loss_none = SoftmaxFocalLoss(
            gamma=2.0, reduction="none", background_class=BG
        )(logits, targets)
        loss_mp = SoftmaxFocalLoss(
            gamma=2.0, reduction="mean_positive", background_class=BG
        )(logits, targets)

        expected = loss_none.sum() / 10
        assert torch.allclose(loss_mp, expected, atol=1e-6)

    def test_mean_positive_sequence(self):
        torch.manual_seed(SEED)
        N, C, L = 4, 5, 64
        logits = torch.randn(N, C, L)
        targets = torch.zeros(N, L, dtype=torch.long)
        positive_positions = [(0, 5), (0, 20), (1, 10), (2, 30), (3, 2)]
        for b, t in positive_positions:
            targets[b, t] = torch.randint(1, C, (1,)).item()
        seq_lens = [64, 48, 40, 16]
        for i, sl in enumerate(seq_lens):
            targets[i, sl:] = -100

        loss = SoftmaxFocalLoss(
            gamma=2.0, reduction="mean_positive", ignore_index=-100, background_class=0
        )(logits, targets)
        loss_none = SoftmaxFocalLoss(
            gamma=2.0, reduction="none", ignore_index=-100, background_class=0
        )(logits, targets)

        expected = loss_none.sum() / len(positive_positions)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_gradient_flow_mean_positive(self):
        torch.manual_seed(SEED)
        N, C = 16, 4
        logits = torch.randn(N, C, requires_grad=True)
        targets = torch.zeros(N, dtype=torch.long)
        targets[:4] = torch.randint(1, C, (4,))
        targets[12:] = -100

        loss = SoftmaxFocalLoss(
            gamma=2.0, reduction="mean_positive", ignore_index=-100, background_class=0
        )(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert (logits.grad[12:] == 0).all()
        assert logits.grad[:12].abs().sum() > 0


# ===========================================================================
# 7. SoftmaxFocalLoss – input shapes
# ===========================================================================


class TestShapes:
    def test_2d_standard(self):
        torch.manual_seed(SEED)
        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean")(
            torch.randn(16, 5), torch.randint(0, 5, (16,))
        )
        assert loss.ndim == 0

    def test_3d_sequence(self):
        torch.manual_seed(SEED)
        N, C, L = 4, 10, 64
        logits = torch.randn(N, C, L)
        targets = torch.randint(0, C, (N, L))
        seq_lens = [64, 48, 32, 16]
        for i, sl in enumerate(seq_lens):
            targets[i, sl:] = -100

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)(logits, targets)
        total_valid = sum(seq_lens)
        loss_none = SoftmaxFocalLoss(gamma=2.0, reduction="none", ignore_index=-100)(logits, targets)
        assert torch.allclose(loss, loss_none.sum() / total_valid, atol=1e-5)

    def test_4d_spatial(self):
        torch.manual_seed(SEED)
        N, C, H, W = 2, 6, 16, 16
        logits = torch.randn(N, C, H, W)
        targets = torch.randint(0, C, (N, H, W))
        targets[:, H // 2:, :] = -100

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)(logits, targets)
        n_valid = N * (H // 2) * W
        loss_none = SoftmaxFocalLoss(gamma=2.0, reduction="none", ignore_index=-100)(logits, targets)
        assert torch.allclose(loss, loss_none.sum() / n_valid, atol=1e-5)


# ===========================================================================
# 8. SoftmaxFocalLoss – gradient flow
# ===========================================================================


class TestGradients:
    def test_basic_gradient_flow(self):
        torch.manual_seed(SEED)
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 4, (16,))
        loss = SoftmaxFocalLoss(gamma=2.0, alpha=[0.25] * 4, reduction="mean")(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert logits.grad.abs().sum() > 0

    def test_padded_positions_zero_grad(self):
        torch.manual_seed(SEED)
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 4, (16,))
        targets[12:] = -100

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)(logits, targets)
        loss.backward()

        assert (logits.grad[12:] == 0).all()
        assert logits.grad[:12].abs().sum() > 0

    def test_all_padded_zero_grad(self):
        logits = torch.randn(8, 4, requires_grad=True)
        targets = torch.full((8,), -100, dtype=torch.long)

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)(logits, targets)
        loss.backward()
        assert (logits.grad == 0).all()
        assert not torch.isnan(logits.grad).any()

    def test_sequence_padding_zero_grad(self):
        torch.manual_seed(SEED)
        N, C, L = 4, 8, 32
        logits = torch.randn(N, C, L, requires_grad=True)
        targets = torch.randint(0, C, (N, L))
        seq_lens = [32, 24, 16, 8]
        for i, sl in enumerate(seq_lens):
            targets[i, sl:] = -100

        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean", ignore_index=-100)(logits, targets)
        loss.backward()

        for i, sl in enumerate(seq_lens):
            if sl < L:
                assert (logits.grad[i, :, sl:] == 0).all()


# ===========================================================================
# 9. SoftmaxFocalLoss – numerical edge cases
# ===========================================================================


class TestNumericalEdgeCases:
    def test_very_large_logits(self):
        N, C = 8, 4
        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean")(
            torch.randn(N, C) * 100, torch.randint(0, C, (N,))
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_very_small_logits(self):
        N, C = 8, 4
        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean")(
            torch.randn(N, C) * 1e-6, torch.randint(0, C, (N,))
        )
        assert not torch.isnan(loss)

    def test_single_sample(self):
        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean")(
            torch.randn(1, 4), torch.randint(0, 4, (1,))
        )
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_two_classes(self):
        loss = SoftmaxFocalLoss(gamma=2.0, alpha=[0.75, 0.25], reduction="mean")(
            torch.randn(32, 2), torch.randint(0, 2, (32,))
        )
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_many_classes(self):
        loss = SoftmaxFocalLoss(gamma=2.0, reduction="mean")(
            torch.randn(16, 1000), torch.randint(0, 1000, (16,))
        )
        assert not torch.isnan(loss)


# ===========================================================================
# 10. SoftmaxFocalLoss – label smoothing
# ===========================================================================


class TestLabelSmoothing:
    def test_label_smoothing_changes_loss(self):
        torch.manual_seed(SEED)
        N, C = 32, 5
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        loss_no = SoftmaxFocalLoss(gamma=2.0, label_smoothing=0.0, reduction="mean")(logits, targets)
        loss_ls = SoftmaxFocalLoss(gamma=2.0, label_smoothing=0.1, reduction="mean")(logits, targets)
        assert not torch.allclose(loss_no, loss_ls, atol=1e-4)

    def test_gamma_zero_label_smoothing_matches_ce(self):
        torch.manual_seed(SEED)
        N, C = 32, 5
        logits = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        focal = SoftmaxFocalLoss(gamma=0.0, label_smoothing=0.1, reduction="mean")(logits, targets)
        ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="mean")(logits, targets)
        assert torch.allclose(focal, ce, atol=1e-6)


# ===========================================================================
# 11. gather_distributed – DDP resolve / no-op at world_size=1
# ===========================================================================


class TestGatherDistributed:
    @pytest.fixture(autouse=True)
    def setup_dist(self):
        _init_single_process_group()
        yield
        _destroy_process_group()

    # --- SigmoidFocalLoss ---------------------------------------------------

    def test_sigmoid_auto_resolves_false_at_world_size_1(self):
        fn = SigmoidFocalLoss()
        assert fn._gather_resolved is None
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 2, (16, 4)).float()
        fn(logits, targets)
        assert fn._gather_resolved is False

    def test_sigmoid_explicit_false_resolves_false(self):
        fn = SigmoidFocalLoss(gather_distributed=False)
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 2, (16, 4)).float()
        fn(logits, targets)
        assert fn._gather_resolved is False

    def test_sigmoid_output_matches_no_gather_at_world_size_1(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 4, requires_grad=True)
        targets = torch.randint(0, 2, (32, 4)).float()
        fn_auto = SigmoidFocalLoss(reduction="mean")
        fn_off  = SigmoidFocalLoss(reduction="mean", gather_distributed=False)
        assert torch.allclose(fn_auto(logits, targets), fn_off(logits, targets))

    def test_sigmoid_gradient_flows_with_gather_flag(self):
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 2, (16, 4)).float()
        fn = SigmoidFocalLoss(reduction="mean")
        fn(logits, targets).backward()
        assert logits.grad is not None

    # --- SoftmaxFocalLoss ---------------------------------------------------

    def test_softmax_auto_resolves_false_at_world_size_1(self):
        fn = SoftmaxFocalLoss()
        assert fn._gather_resolved is None
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 4, (16,))
        fn(logits, targets)
        assert fn._gather_resolved is False

    def test_softmax_explicit_false_resolves_false(self):
        fn = SoftmaxFocalLoss(gather_distributed=False)
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 4, (16,))
        fn(logits, targets)
        assert fn._gather_resolved is False

    def test_softmax_gather_resolved_cached(self):
        fn = SoftmaxFocalLoss()
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 4, (16,))
        fn(logits, targets)
        first = fn._gather_resolved
        fn(logits.detach().requires_grad_(True), targets)
        assert fn._gather_resolved is first

    def test_softmax_output_matches_no_gather_at_world_size_1(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 4, requires_grad=True)
        targets = torch.randint(0, 4, (32,))
        fn_auto = SoftmaxFocalLoss()
        fn_off  = SoftmaxFocalLoss(gather_distributed=False)
        assert torch.allclose(fn_auto(logits, targets), fn_off(logits, targets))

    def test_softmax_gradient_flows_with_gather_flag(self):
        logits = torch.randn(32, 4, requires_grad=True)
        targets = torch.randint(0, 4, (32,))
        fn = SoftmaxFocalLoss()
        fn(logits, targets).backward()
        assert logits.grad is not None
