"""Tests for RL loss functions."""

import pytest
import torch

from ml_interview.losses import entropy_bonus, ppo_clip_loss, reinforce_loss, value_loss


class TestReinforceLoss:
    def test_positive_reward_negative_loss(self):
        log_probs = torch.tensor([-1.0, -0.5, -2.0])
        rewards = torch.tensor([1.0, 1.0, 1.0])
        loss = reinforce_loss(log_probs, rewards)
        # Loss = -mean(log_probs * rewards) > 0
        assert loss.item() > 0

    def test_with_baseline(self):
        log_probs = torch.tensor([-1.0, -0.5])
        rewards = torch.tensor([2.0, 2.0])
        baseline = torch.tensor([1.0, 1.0])
        loss = reinforce_loss(log_probs, rewards, baseline=baseline)
        assert torch.isfinite(loss)

    def test_scalar_output(self):
        log_probs = torch.randn(8)
        rewards = torch.randn(8)
        assert reinforce_loss(log_probs, rewards).shape == torch.Size([])


class TestPPOClipLoss:
    def test_scalar_output(self):
        lp_new = torch.randn(16)
        lp_old = torch.randn(16)
        adv = torch.randn(16)
        assert ppo_clip_loss(lp_new, lp_old, adv).shape == torch.Size([])

    def test_no_update_when_ratio_1(self):
        """If log_probs_new == log_probs_old, ratio=1 and loss = -mean(adv)."""
        log_probs = torch.zeros(4)
        advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = ppo_clip_loss(log_probs, log_probs, advantages)
        expected = -advantages.mean()
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_finite(self):
        lp_new = torch.log(torch.rand(10) + 1e-6)
        lp_old = torch.log(torch.rand(10) + 1e-6)
        adv = torch.randn(10)
        assert torch.isfinite(ppo_clip_loss(lp_new, lp_old, adv))


class TestValueLoss:
    def test_perfect_prediction_zero_loss(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        r = torch.tensor([1.0, 2.0, 3.0])
        assert value_loss(v, r).item() == pytest.approx(0.0)

    def test_loss_positive(self):
        v = torch.tensor([0.0, 0.0])
        r = torch.tensor([1.0, 2.0])
        assert value_loss(v, r).item() > 0

    def test_output_shape(self):
        assert value_loss(torch.randn(8), torch.randn(8)).shape == torch.Size([])


class TestEntropyBonus:
    def test_uniform_distribution_max_entropy(self):
        vocab = 4
        logits = torch.zeros(1, vocab)  # uniform after softmax
        entropy = entropy_bonus(logits)
        import math
        expected = math.log(vocab)
        assert entropy.item() == pytest.approx(expected, rel=1e-4)

    def test_entropy_non_negative(self):
        logits = torch.randn(8, 100)
        assert entropy_bonus(logits).item() >= 0

    def test_scalar_output(self):
        assert entropy_bonus(torch.randn(4, 50)).shape == torch.Size([])
