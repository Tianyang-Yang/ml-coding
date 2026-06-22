"""Transformer block components for ML interview practice."""

import torch
import torch.nn as nn

from ml_interview.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network (two linear layers + ReLU).

    Args:
        d_model: Input and output dimension.
        d_ff: Hidden dimension (typically 4 * d_model).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block (Pre-LN variant).

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with pre-layer normalisation.

        Args:
            x:    (batch, seq, d_model)
            mask: Optional attention mask.

        Returns:
            Tensor of shape (batch, seq, d_model).
        """
        # Self-attention with residual connection
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # Feed-forward with residual connection
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block (Pre-LN variant).

    Includes masked self-attention and cross-attention sub-layers.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:           (batch, tgt_seq, d_model) — decoder input.
            memory:      (batch, src_seq, d_model) — encoder output.
            tgt_mask:    Optional causal mask for self-attention.
            memory_mask: Optional padding mask for cross-attention.

        Returns:
            Tensor of shape (batch, tgt_seq, d_model).
        """
        n1 = self.norm1(x)
        x = x + self.drop(self.self_attn(n1, n1, n1, tgt_mask))
        n2 = self.norm2(x)
        x = x + self.drop(self.cross_attn(n2, memory, memory, memory_mask))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x
