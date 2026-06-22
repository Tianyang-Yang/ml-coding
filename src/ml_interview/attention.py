"""Attention mechanisms for ML interview practice."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch, heads, seq_q, d_k).
        K: Key tensor of shape (batch, heads, seq_k, d_k).
        V: Value tensor of shape (batch, heads, seq_k, d_v).
        mask: Optional boolean mask of shape broadcastable to
              (batch, heads, seq_q, seq_k). ``True`` means *ignore* that
              position (logit set to -inf before softmax).

    Returns:
        Output tensor of shape (batch, heads, seq_q, d_v).
    """
    d_k = Q.size(-1)
    # (batch, heads, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def make_causal_mask(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Return a causal boolean mask of shape (T, T).

    ``mask[i, j] == True`` means position *i* should **not** attend to
    position *j* (i.e. j > i — future token).
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.

    Args:
        d_model: Model (embedding) dimension.
        num_heads: Number of attention heads. ``d_model`` must be divisible
                   by ``num_heads``.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, d_model) → (batch, heads, seq, d_k)."""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, heads, seq, d_k) → (batch, seq, d_model)."""
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  Optional boolean mask broadcastable to
                   (batch, heads, seq_q, seq_k).

        Returns:
            Output tensor of shape (batch, seq_q, d_model).
        """
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        merged = self._merge_heads(attn_out)
        return self.W_o(merged)
