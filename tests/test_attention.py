"""Tests for attention mechanisms."""

import math

import pytest
import torch

from ml_interview.attention import (
    MultiHeadAttention,
    make_causal_mask,
    scaled_dot_product_attention,
)


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------


class TestScaledDotProductAttention:
    def _random_qkv(self, batch=2, heads=4, seq_q=6, seq_k=8, d_k=16, d_v=32):
        Q = torch.randn(batch, heads, seq_q, d_k)
        K = torch.randn(batch, heads, seq_k, d_k)
        V = torch.randn(batch, heads, seq_k, d_v)
        return Q, K, V

    def test_output_shape(self):
        Q, K, V = self._random_qkv()
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 4, 6, 32)

    def test_no_mask(self):
        Q, K, V = self._random_qkv(seq_q=4, seq_k=4, d_k=8, d_v=8)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape[-1] == 8

    def test_identity_attention(self):
        """When Q == K == V == I (and d_k=1), output should approximately equal V."""
        T = 5
        Q = torch.ones(1, 1, T, 1)
        K = torch.ones(1, 1, T, 1)
        V = torch.eye(T).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (1, 1, T, T)

    def test_boolean_mask_applied(self):
        """Masked positions should have zero contribution (softmax → 0)."""
        batch, heads, seq, d_k = 1, 1, 4, 8
        Q = torch.randn(batch, heads, seq, d_k)
        K = torch.randn(batch, heads, seq, d_k)
        V = torch.randn(batch, heads, seq, d_k)

        # Mask all positions except the first
        mask = torch.ones(1, 1, seq, seq, dtype=torch.bool)
        mask[:, :, :, 0] = False  # allow attending to position 0

        out_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
        # Output at every query position should equal V[:, :, 0, :]
        expected = V[:, :, 0:1, :].expand_as(out_masked)
        assert torch.allclose(out_masked, expected, atol=1e-5)

    def test_scaling_factor(self):
        """Scores should be divided by sqrt(d_k)."""
        d_k = 64
        Q = torch.ones(1, 1, 1, d_k)
        K = torch.ones(1, 1, 1, d_k)
        V = torch.ones(1, 1, 1, d_k)
        out = scaled_dot_product_attention(Q, K, V)
        # With Q=K=V=1-filled, result should just be the value
        assert out.shape == (1, 1, 1, d_k)


# ---------------------------------------------------------------------------
# make_causal_mask
# ---------------------------------------------------------------------------


class TestMakeCausalMask:
    def test_shape(self):
        mask = make_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_dtype_bool(self):
        mask = make_causal_mask(4)
        assert mask.dtype == torch.bool

    def test_upper_triangular_true(self):
        mask = make_causal_mask(5)
        # Future positions (j > i) should be True
        for i in range(5):
            for j in range(5):
                expected = j > i
                assert mask[i, j].item() == expected

    def test_no_masking_for_t1(self):
        mask = make_causal_mask(1)
        assert not mask[0, 0].item()


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        out = mha(x, x, x)
        assert out.shape == (2, 10, 64)

    def test_cross_attention_shape(self):
        mha = MultiHeadAttention(d_model=32, num_heads=4)
        q = torch.randn(2, 6, 32)
        kv = torch.randn(2, 10, 32)
        out = mha(q, kv, kv)
        assert out.shape == (2, 6, 32)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(d_model=64, num_heads=7)

    def test_causal_mask_integration(self):
        d_model, num_heads, seq = 32, 4, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(1, seq, d_model)
        mask = make_causal_mask(seq)
        # Should not raise; output shape must be correct
        out = mha(x, x, x, mask=mask.unsqueeze(0).unsqueeze(0))
        assert out.shape == (1, seq, d_model)

    def test_parameter_count(self):
        d_model, num_heads = 64, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        n_params = sum(p.numel() for p in mha.parameters())
        # 4 weight matrices each (d_model × d_model), no bias
        assert n_params == 4 * d_model * d_model
