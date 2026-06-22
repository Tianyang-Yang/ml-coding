"""Key-value cache for efficient autoregressive inference."""

from __future__ import annotations

import torch


class KVCache:
    """Incremental key-value cache for transformer inference.

    Stores past keys and values for each layer so that only the new token's
    projections need to be computed at each decoding step.

    Args:
        num_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length supported.
        num_heads: Number of attention heads.
        d_k: Head dimension (d_model // num_heads).
        device: Torch device.
        dtype: Tensor dtype (default: float32).
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        d_k: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self._keys: list[torch.Tensor] = [
            torch.zeros(1, num_heads, max_seq_len, d_k, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self._values: list[torch.Tensor] = [
            torch.zeros(1, num_heads, max_seq_len, d_k, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self._length: int = 0

    @property
    def length(self) -> int:
        """Number of tokens currently stored in the cache."""
        return self._length

    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full accumulated tensors.

        Args:
            layer_idx: Index of the transformer layer (0-based).
            new_keys:   (batch, heads, new_seq, d_k)
            new_values: (batch, heads, new_seq, d_k)

        Returns:
            Tuple of (keys, values) each of shape
            (batch, heads, current_length + new_seq, d_k).
        """
        new_len = new_keys.size(2)
        end = self._length + new_len
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KVCache overflow: current length {self._length} + new tokens "
                f"{new_len} exceeds max_seq_len {self.max_seq_len}"
            )
        self._keys[layer_idx][:, :, self._length : end, :] = new_keys
        self._values[layer_idx][:, :, self._length : end, :] = new_values
        # Only update length once (caller is expected to call all layers per step)
        if layer_idx == self.num_layers - 1:
            self._length = end
        keys = self._keys[layer_idx][:, :, :end, :]
        values = self._values[layer_idx][:, :, :end, :]
        return keys, values

    def reset(self) -> None:
        """Clear the cache (zero out stored tensors and reset length)."""
        for k, v in zip(self._keys, self._values):
            k.zero_()
            v.zero_()
        self._length = 0
