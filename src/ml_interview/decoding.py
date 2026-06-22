"""Decoding strategies for autoregressive language models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy_decode(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Select the token with the highest probability at each step.

    Args:
        logits: (batch, vocab_size) — unnormalised scores for the next token.

    Returns:
        (batch,) tensor of selected token ids.
    """
    return logits.argmax(dim=-1)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k.

    Args:
        logits: (batch, vocab_size)
        k: Number of top tokens to keep.

    Returns:
        Filtered logits of the same shape with non-top-k values set to -inf.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    top_k_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_vals[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep the smallest set of tokens whose
    cumulative probability exceeds *p*.

    Args:
        logits: (batch, vocab_size)
        p: Cumulative probability threshold in (0, 1].

    Returns:
        Filtered logits with tokens outside the nucleus set to -inf.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens whose cumulative probability exceeds p (shift by 1 to
    # keep at least one token).
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    # Scatter filtered values back to original ordering
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample the next token from a distribution defined by *logits*.

    Args:
        logits:      (batch, vocab_size) — unnormalised scores.
        temperature: Softmax temperature. Values < 1 make the distribution
                     sharper; values > 1 make it flatter.
        top_k:       If > 0, apply top-k filtering before sampling.
        top_p:       If < 1.0, apply nucleus filtering before sampling.

    Returns:
        (batch,) tensor of sampled token ids.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = logits / temperature
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
