# 003 — Causal (Auto-regressive) Mask

## Prompt

Write a function that generates a causal attention mask for a sequence of length
`T`. Position `i` should only attend to positions `j ≤ i`.

**Signature**

```python
def make_causal_mask(T: int, device: torch.device = "cpu") -> torch.Tensor:
    """
    Returns a boolean mask of shape (T, T) where mask[i, j] is True
    when position i should NOT attend to position j (i.e. j > i).
    """
    ...
```

Then show how you would pass this mask into `scaled_dot_product_attention`.

## Constraints

- Return a **boolean** tensor (`torch.bool`), not float.
- Use only tensor operations; no Python-level loops.

## Follow-up Questions

1. Why is a causal mask necessary for decoder-only language models?
2. How does the causal mask interact with padding masks?
3. For a sequence of length T, how many positions are masked (set to `-inf`) on
   average per query position?
4. How would you implement a sliding-window (local) attention mask?

## Reference

- Vaswani et al., *Attention Is All You Need* (2017) — Section 3.1 (decoder)
