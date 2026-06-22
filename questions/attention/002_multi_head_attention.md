# 002 — Multi-Head Attention

## Prompt

Implement a `MultiHeadAttention` module in PyTorch using your
`scaled_dot_product_attention` from question 001.

**Signature**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        ...

    def forward(
        self,
        query: torch.Tensor,   # (batch, seq_q, d_model)
        key:   torch.Tensor,   # (batch, seq_k, d_model)
        value: torch.Tensor,   # (batch, seq_k, d_model)
        mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:         # (batch, seq_q, d_model)
        ...
```

## Constraints

- `d_model` must be divisible by `num_heads`.
- Use four linear projections: Wq, Wk, Wv, Wo (no bias required).
- Do **not** use `nn.MultiheadAttention`.

## Follow-up Questions

1. Why use multiple heads rather than one large head?
2. How does the parameter count compare to a single-head equivalent?
3. What is the role of the output projection Wo?
4. How would you add relative positional encodings (RoPE) to this module?

## Reference

- Vaswani et al., *Attention Is All You Need* (2017) — Section 3.2.2
