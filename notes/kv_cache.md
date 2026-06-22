# KV Cache — Study Notes

## Motivation

During autoregressive generation, the keys and values for all past tokens are
recomputed at every step without caching — O(n²) total compute.

A KV cache stores those tensors so each new token only needs to compute
*one* new key/value pair — O(n) total compute.

## How It Works

1. Allocate fixed buffers `keys[layer]`, `values[layer]` of shape
   `(batch, heads, max_seq_len, d_k)`.
2. At step *t*, compute K_t and V_t for the new token.
3. Write K_t, V_t into position *t* of the buffer.
4. Read back `keys[:, :, :t+1, :]` and `values[:, :, :t+1, :]` for attention.

## Memory Cost

`2 × num_layers × num_heads × d_k × max_seq_len × batch_size × bytes_per_element`

For GPT-3 175B at fp16: ~350 GB for batch_size=1, seq_len=2048.

## Optimisations

| Technique | Idea |
|---|---|
| Multi-Query Attention (MQA) | Share K/V across all heads |
| Grouped-Query Attention (GQA) | Share K/V across groups of heads |
| Sliding Window | Only keep last *w* tokens in cache |
| PagedAttention (vLLM) | Manage cache memory in pages like virtual memory |

## Common Interview Pitfalls

- Forgetting to handle the batch dimension.
- Not accounting for the cache in memory footprint estimates.
- Applying the causal mask incorrectly when the cache is active
  (query position is the current step, key positions are 0..t).
