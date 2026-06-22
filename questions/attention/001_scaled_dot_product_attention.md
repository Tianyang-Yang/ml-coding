# 001 — Scaled Dot-Product Attention

## Prompt

Implement scaled dot-product attention from scratch in PyTorch **without** using
`torch.nn.functional.scaled_dot_product_attention`.

```
Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V
```

**Signature**

```python
def scaled_dot_product_attention(
    Q: torch.Tensor,   # (batch, heads, seq_q, d_k)
    K: torch.Tensor,   # (batch, heads, seq_k, d_k)
    V: torch.Tensor,   # (batch, heads, seq_k, d_v)
    mask: torch.Tensor | None = None,  # broadcastable bool mask (True = ignore)
) -> torch.Tensor:     # (batch, heads, seq_q, d_v)
    ...
```

## Constraints

- Do **not** use any built-in attention helpers.
- Handle an optional boolean mask where `True` means the position should be
  masked (set logit to `-inf` before softmax).
- The function must be numerically stable (no explicit `exp` overflow).

## Follow-up Questions

1. Why do we divide by √dₖ? What goes wrong without it?
2. How does the complexity scale with sequence length?
3. What is the difference between additive and multiplicative attention?
4. How would you implement a numerically stable softmax?

## Reference

- Vaswani et al., *Attention Is All You Need* (2017) — Section 3.2.1
