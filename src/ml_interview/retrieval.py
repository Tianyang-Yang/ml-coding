"""Dense retrieval and similarity utilities for ML interview practice."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise cosine similarity between two sets of vectors.

    Args:
        a: (n, d) — query embeddings.
        b: (m, d) — corpus embeddings.

    Returns:
        (n, m) matrix where entry [i, j] is cos_sim(a[i], b[j]).
    """
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return torch.matmul(a_norm, b_norm.T)


def dot_product_similarity_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise dot-product similarity.

    Args:
        a: (n, d) — query embeddings.
        b: (m, d) — corpus embeddings.

    Returns:
        (n, m) similarity matrix.
    """
    return torch.matmul(a, b.T)


def top_k_retrieval(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    k: int,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve the top-k corpus items for each query.

    Args:
        query_embeddings:  (n_queries, d)
        corpus_embeddings: (n_corpus, d)
        k: Number of items to retrieve per query.
        normalize: If True use cosine similarity, otherwise dot product.

    Returns:
        Tuple of (scores, indices), each of shape (n_queries, k).
    """
    if normalize:
        sim = cosine_similarity_matrix(query_embeddings, corpus_embeddings)
    else:
        sim = dot_product_similarity_matrix(query_embeddings, corpus_embeddings)
    scores, indices = torch.topk(sim, k=k, dim=-1)
    return scores, indices
