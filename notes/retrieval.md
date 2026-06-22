# Retrieval — Study Notes

## Dense Retrieval

Encode queries and documents into dense vectors; retrieve by approximate or
exact nearest-neighbour search.

**Key models:** DPR, Contriever, E5, BGE, OpenAI text-embedding-*.

### Training Objective (Contrastive)

```
L = -log [exp(q · d⁺ / τ) / Σ_d exp(q · d / τ)]
```

In-batch negatives make this efficient at scale.

## Similarity Functions

| Function | Formula | Notes |
|---|---|---|
| Cosine | (a · b) / (|a| |b|) | Scale-invariant; normalise embeddings first |
| Dot product | a · b | Magnitude matters; can use FAISS directly |
| L2 | -|a - b|² | Equivalent to dot product when normalised |

## Approximate Nearest Neighbour (ANN)

- **FAISS (Facebook AI Similarity Search):** IVF, HNSW, PQ indices.
- **Annoy:** Random-projection trees — good for read-heavy workloads.
- **ScaNN:** Google's SIMD-optimised ANN.

## Retrieval Pipeline

```
Query → Encoder → ANN Index → Top-k docs → Re-ranker → Final ranking
```

## Common Interview Pitfalls

- Comparing embeddings from different models (incompatible spaces).
- Forgetting that cosine similarity requires L2-normalised vectors.
- Not updating the index after adding new documents.
