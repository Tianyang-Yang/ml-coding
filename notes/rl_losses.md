# RL Losses — Study Notes

## REINFORCE (Policy Gradient)

```
∇θ J(θ) ≈ Eπ[∇θ log π_θ(a|s) · R]
```

**Loss:** `L = -E[log π(a|s) · (R - b)]`  
**Baseline** *b* (e.g. value function) reduces variance without adding bias.

**Pros:** Simple, unbiased.  
**Cons:** High variance, slow convergence.

## PPO (Proximal Policy Optimization)

Clip the probability ratio to avoid large policy updates:

```
r_t = π_new(a|s) / π_old(a|s)
L_CLIP = -E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
```

**ε** is typically 0.2.  
**Advantages** A_t are usually normalised (zero mean, unit variance) per batch.

## Value Loss

```
L_V = E[(V(s) - G_t)²]
```

Trains the critic to predict returns.

## Entropy Bonus

```
L_entropy = -E[Σ_a π(a|s) log π(a|s)]
```

Added to the loss (with small coefficient) to encourage exploration.

## Full PPO Objective

```
L = L_CLIP - c₁ · L_V + c₂ · L_entropy
```

## Common Interview Pitfalls

- Forgetting to `.detach()` old log-probs and advantages.
- Not normalising advantages.
- Using raw rewards instead of discounted returns G_t.
- Clipping too aggressively (small ε) → policy barely updates.
