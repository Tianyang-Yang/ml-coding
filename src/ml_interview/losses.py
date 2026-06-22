"""Reinforcement-learning loss functions for ML interview practice."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """REINFORCE / policy-gradient loss (Williams, 1992).

    Loss = -E[log π(a|s) · (R - b)]

    Args:
        log_probs: (batch,) log-probabilities of the taken actions.
        rewards:   (batch,) observed returns / rewards.
        baseline:  (batch,) optional value-function baseline to reduce variance.

    Returns:
        Scalar loss tensor.
    """
    advantage = rewards if baseline is None else rewards - baseline
    return -(log_probs * advantage.detach()).mean()


def ppo_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate objective (Schulman et al., 2017).

    L_CLIP = -E[min(r·A, clip(r, 1-ε, 1+ε)·A)]

    where r = π_new(a|s) / π_old(a|s).

    Args:
        log_probs_new: (batch,) log-probs under current policy.
        log_probs_old: (batch,) log-probs under old policy (detached).
        advantages:    (batch,) advantage estimates (should be normalised).
        clip_epsilon:  Clipping parameter ε (default 0.2).

    Returns:
        Scalar loss tensor (to be minimised).
    """
    ratio = torch.exp(log_probs_new - log_probs_old.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate1 = ratio * advantages.detach()
    surrogate2 = clipped_ratio * advantages.detach()
    return -torch.min(surrogate1, surrogate2).mean()


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared-error value function loss.

    Args:
        values:  (batch,) predicted state values V(s).
        returns: (batch,) discounted returns G_t.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(values, returns.detach())


def entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of a categorical policy — used as an exploration bonus.

    Args:
        logits: (batch, vocab_size) — unnormalised action logits.

    Returns:
        Mean entropy scalar (positive; add to loss with a small coefficient
        to encourage exploration).
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()
