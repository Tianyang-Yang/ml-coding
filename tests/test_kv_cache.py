"""Tests for KV cache."""

import pytest
import torch

from ml_interview.kv_cache import KVCache


class TestKVCache:
    def _make_cache(self, num_layers=2, max_seq_len=16, num_heads=4, d_k=8):
        return KVCache(
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            d_k=d_k,
        )

    def test_initial_length(self):
        cache = self._make_cache()
        assert cache.length == 0

    def test_update_returns_correct_shape(self):
        cache = self._make_cache(num_layers=2, num_heads=4, d_k=8, max_seq_len=16)
        new_k = torch.randn(1, 4, 3, 8)
        new_v = torch.randn(1, 4, 3, 8)
        # Update both layers (length increments after last layer)
        cache.update(0, new_k, new_v)
        keys, values = cache.update(1, new_k, new_v)
        assert keys.shape == (1, 4, 3, 8)
        assert values.shape == (1, 4, 3, 8)

    def test_length_increments(self):
        cache = self._make_cache(num_layers=1, max_seq_len=16, num_heads=2, d_k=4)
        new_k = torch.randn(1, 2, 5, 4)
        new_v = torch.randn(1, 2, 5, 4)
        cache.update(0, new_k, new_v)
        assert cache.length == 5

    def test_accumulated_keys_correct(self):
        cache = self._make_cache(num_layers=1, max_seq_len=16, num_heads=1, d_k=4)
        k1 = torch.randn(1, 1, 3, 4)
        v1 = torch.randn(1, 1, 3, 4)
        cache.update(0, k1, v1)
        k2 = torch.randn(1, 1, 2, 4)
        v2 = torch.randn(1, 1, 2, 4)
        keys, values = cache.update(0, k2, v2)
        # After second update length becomes 5
        assert keys.shape == (1, 1, 5, 4)
        assert torch.allclose(keys[:, :, :3, :], k1)
        assert torch.allclose(keys[:, :, 3:, :], k2)

    def test_overflow_raises(self):
        cache = self._make_cache(num_layers=1, max_seq_len=4, num_heads=1, d_k=4)
        k = torch.randn(1, 1, 5, 4)
        v = torch.randn(1, 1, 5, 4)
        with pytest.raises(RuntimeError, match="overflow"):
            cache.update(0, k, v)

    def test_reset(self):
        cache = self._make_cache(num_layers=1, max_seq_len=8, num_heads=2, d_k=4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache.update(0, k, v)
        assert cache.length == 3
        cache.reset()
        assert cache.length == 0
