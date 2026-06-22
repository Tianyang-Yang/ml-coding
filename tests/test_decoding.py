"""Tests for decoding strategies."""

import pytest
import torch

from ml_interview.decoding import greedy_decode, sample, top_k_filter, top_p_filter


class TestGreedyDecode:
    def test_selects_argmax(self):
        logits = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.1, 0.4]])
        tokens = greedy_decode(logits)
        assert tokens.tolist() == [1, 0]

    def test_output_shape(self):
        logits = torch.randn(4, 1000)
        assert greedy_decode(logits).shape == (4,)


class TestTopKFilter:
    def test_only_k_values_remain(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = top_k_filter(logits, k=3)
        finite_count = torch.isfinite(filtered).sum().item()
        assert finite_count == 3

    def test_top_values_kept(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_filter(logits, k=2)
        # Top-2 are indices 1 (5.0) and 4 (4.0)
        assert torch.isfinite(filtered[0, 1])
        assert torch.isfinite(filtered[0, 4])
        assert not torch.isfinite(filtered[0, 0])

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            top_k_filter(torch.randn(1, 10), k=0)


class TestTopPFilter:
    def test_output_shape(self):
        logits = torch.randn(2, 50)
        filtered = top_p_filter(logits, p=0.9)
        assert filtered.shape == (2, 50)

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            top_p_filter(torch.randn(1, 10), p=0.0)

    def test_p_equals_one_keeps_all(self):
        logits = torch.randn(1, 10)
        filtered = top_p_filter(logits, p=1.0)
        assert torch.isfinite(filtered).all()


class TestSample:
    def test_output_shape(self):
        logits = torch.randn(4, 100)
        tokens = sample(logits)
        assert tokens.shape == (4,)

    def test_tokens_in_valid_range(self):
        vocab_size = 200
        logits = torch.randn(8, vocab_size)
        tokens = sample(logits)
        assert (tokens >= 0).all() and (tokens < vocab_size).all()

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            sample(torch.randn(1, 10), temperature=0.0)

    def test_with_top_k(self):
        logits = torch.randn(2, 50)
        tokens = sample(logits, top_k=5)
        assert tokens.shape == (2,)

    def test_with_top_p(self):
        logits = torch.randn(2, 50)
        tokens = sample(logits, top_p=0.9)
        assert tokens.shape == (2,)
