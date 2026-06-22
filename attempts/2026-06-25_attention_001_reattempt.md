# 2026-06-25 — Attention 001: Scaled Dot-Product Attention (Re-attempt)

**Question:** `questions/attention/001_scaled_dot_product_attention.md`  
**Date:** 2026-06-25  
**Time taken:** ~8 min (improved from first attempt)

---

## My Implementation

```python
import math, torch, torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V
```

## Improvement vs First Attempt

- Wrote it confidently without hesitation.
- Used `@` operator for clarity.
- Remembered `dim=-1` in softmax immediately.
- No mistakes during the attempt.

## Follow-up Answers (Improved)

1. **Why √dₖ?** Variance of Qᵢ · Kᵢ ≈ dₖ for unit normal Q, K. Dividing
   by √dₖ normalises to variance ≈ 1 → stable gradients through softmax.

2. **Complexity?** O(n²d) time, O(n²) attention matrix space.

3. **Numerically stable softmax?** Subtract `max(x)` before `exp`:
   `softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))`.

4. **Additive vs multiplicative?** Additive: score = vᵀ tanh(Wq·q + Wk·k).
   Multiplicative: score = q·k / √dₖ. Multiplicative is faster in practice.

## Self-Assessment

Much faster and more confident. Ready to move to question 002 (MultiHeadAttention).
