# Attention — Study Notes

## Core Idea

Attention allows a model to weigh different positions of the input sequence
when computing a representation for any given position.

## Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V
```

- **Why √dₖ?** Without scaling, the dot products grow large in magnitude
  (variance ≈ dₖ), pushing softmax into regions of extremely small gradients.
- **Complexity:** O(n² · dₖ) — quadratic in sequence length.

## Multi-Head Attention

Run *h* attention heads in parallel over subspaces of dimension dₖ = d_model/h.
Concatenate outputs then apply Wo.

**Benefits**
- Different heads can attend to different representation subspaces.
- More expressive than a single large attention.

## Causal (Auto-regressive) Mask

For decoder-only LMs, token *i* must not see tokens *j > i*.
Achieved by adding −∞ to future positions before softmax.

## Key Variants

| Variant | Description |
|---|---|
| Full (bidirectional) | All positions attend to all positions |
| Causal | Each position attends to past only |
| Cross-attention | Queries from decoder, Keys/Values from encoder |
| Sparse attention | Only a subset of positions (e.g., local window) |

## Positional Encoding

Original transformer: sinusoidal, added to token embeddings.  
Modern LLMs: **RoPE** (Rotary Position Embedding) applied to Q and K inside each head.

## Common Interview Pitfalls

- Forgetting to scale by 1/√dₖ.
- Applying mask **after** softmax (must be before).
- Not handling the batch dimension correctly when splitting heads.
