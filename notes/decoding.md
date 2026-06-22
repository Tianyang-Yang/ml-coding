# Decoding — Study Notes

## Greedy Decoding

Always pick the token with the highest probability. Fast but suboptimal — tends
to produce repetitive, generic text.

## Beam Search

Keep the *B* most likely partial sequences (beams) at each step. Explore
multiple hypotheses and return the one with the highest overall log-prob.

**Complexity:** O(B · V · T) where V = vocab size, T = output length.

## Sampling Strategies

### Temperature Sampling
Divide logits by temperature τ before softmax.
- τ < 1 → sharper distribution (more confident)
- τ > 1 → flatter distribution (more random)

### Top-k Sampling
Restrict sampling to the k tokens with the highest probability.

### Top-p (Nucleus) Sampling
Restrict sampling to the smallest set of tokens whose cumulative probability
exceeds p. Adaptive — uses more tokens when the distribution is flat.

### Typical Sampling
Sample from tokens whose probability is close to the distribution's entropy.
Tends to produce "humanlike" continuations.

## Common Interview Pitfalls

- Applying top-k/top-p **after** temperature scaling (correct order matters).
- Forgetting that log-probs can underflow with long sequences in beam search
  → use log-space arithmetic.
- Confusing beam search "score" (length-normalised log-prob) with raw log-prob.

## Practical Tips

- For factual tasks: low temperature + greedy or small beam.
- For creative tasks: temperature 0.7–1.2 + top-p 0.9.
