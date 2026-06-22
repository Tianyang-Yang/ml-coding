# Common Mistakes Log

Track recurring mistakes here to avoid repeating them.

---

## Attention

- [ ] **Off-by-one on head dimension:** `d_k = d_model / num_heads`, not `d_model`.
- [ ] **Mask applied after softmax** — must mask logits *before* softmax.
- [ ] **Transpose confusion:** `Q @ K.transpose(-2, -1)`, not `K.T`.

## Transformer

- [ ] **Residual connection skipped** in encoder/decoder sub-layer.
- [ ] **LayerNorm placement:** Pre-LN (norm before sub-layer) vs Post-LN (norm after).

## Decoding

- [ ] **Temperature applied after top-k/top-p** — should be before.
- [ ] **`torch.multinomial` requires probabilities, not logits** — apply softmax first.

## KV Cache

- [ ] **Causal mask not updated** when key/value length grows beyond query length.
- [ ] **Length counter not incremented** after each decode step.

## Metrics

- [ ] **NDCG denominator:** `log2(rank + 1)`, not `log2(rank)`.
- [ ] **AP formula:** divide accumulated precision by *total* relevant, not retrieved.

## RL Losses

- [ ] **Advantages not detached** in PPO — leads to training instability.
- [ ] **Using raw rewards instead of returns** in value loss.
