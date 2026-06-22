"""Tests for transformer block components."""

import torch
import pytest

from ml_interview.transformer import FeedForward, TransformerDecoderBlock, TransformerEncoderBlock


class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(d_model=64, d_ff=256)
        x = torch.randn(2, 10, 64)
        assert ff(x).shape == (2, 10, 64)


class TestTransformerEncoderBlock:
    def test_output_shape(self):
        block = TransformerEncoderBlock(d_model=64, num_heads=8, d_ff=256)
        x = torch.randn(2, 12, 64)
        out = block(x)
        assert out.shape == (2, 12, 64)

    def test_with_mask(self):
        block = TransformerEncoderBlock(d_model=32, num_heads=4, d_ff=128)
        x = torch.randn(1, 6, 32)
        mask = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
        out = block(x, mask=mask)
        assert out.shape == (1, 6, 32)

    def test_residual_connection_shape_preserved(self):
        """Input and output shapes should match."""
        block = TransformerEncoderBlock(d_model=16, num_heads=2, d_ff=64)
        x = torch.randn(3, 5, 16)
        assert block(x).shape == x.shape


class TestTransformerDecoderBlock:
    def test_output_shape(self):
        block = TransformerDecoderBlock(d_model=64, num_heads=8, d_ff=256)
        tgt = torch.randn(2, 8, 64)
        mem = torch.randn(2, 12, 64)
        out = block(tgt, mem)
        assert out.shape == (2, 8, 64)

    def test_with_masks(self):
        block = TransformerDecoderBlock(d_model=32, num_heads=4, d_ff=128)
        tgt = torch.randn(1, 5, 32)
        mem = torch.randn(1, 7, 32)
        tgt_mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
        mem_mask = torch.zeros(1, 1, 5, 7, dtype=torch.bool)
        out = block(tgt, mem, tgt_mask=tgt_mask, memory_mask=mem_mask)
        assert out.shape == (1, 5, 32)
