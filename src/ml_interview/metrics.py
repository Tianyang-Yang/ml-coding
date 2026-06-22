"""Ranking and retrieval evaluation metrics for ML interview practice."""

from __future__ import annotations

import torch


def precision_at_k(relevant: torch.Tensor, retrieved: torch.Tensor, k: int) -> float:
    """Precision@K — fraction of top-K retrieved items that are relevant.

    Args:
        relevant:  1-D boolean tensor of shape (n_corpus,); True = relevant.
        retrieved: 1-D integer tensor of retrieved item indices (ordered).
        k: Cut-off rank.

    Returns:
        Scalar precision value in [0, 1].
    """
    top_k = retrieved[:k]
    hits = relevant[top_k].sum().item()
    return hits / k


def recall_at_k(relevant: torch.Tensor, retrieved: torch.Tensor, k: int) -> float:
    """Recall@K — fraction of all relevant items found in top-K.

    Args:
        relevant:  1-D boolean tensor (True = relevant).
        retrieved: 1-D integer tensor of retrieved item indices (ordered).
        k: Cut-off rank.

    Returns:
        Scalar recall value in [0, 1]. Returns 0 if no relevant items exist.
    """
    n_relevant = relevant.sum().item()
    if n_relevant == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = relevant[top_k].sum().item()
    return hits / n_relevant


def reciprocal_rank(relevant: torch.Tensor, retrieved: torch.Tensor) -> float:
    """Reciprocal rank of the first relevant item in the ranked list.

    Args:
        relevant:  1-D boolean tensor (True = relevant).
        retrieved: 1-D integer tensor of retrieved item indices (ordered).

    Returns:
        1 / rank of first relevant hit, or 0.0 if none found.
    """
    for rank, idx in enumerate(retrieved.tolist(), start=1):
        if relevant[idx]:
            return 1.0 / rank
    return 0.0


def average_precision(relevant: torch.Tensor, retrieved: torch.Tensor) -> float:
    """Average Precision (AP) for a single query.

    Args:
        relevant:  1-D boolean tensor (True = relevant).
        retrieved: 1-D integer tensor of retrieved item indices (ordered).

    Returns:
        AP score in [0, 1].
    """
    n_relevant = relevant.sum().item()
    if n_relevant == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for rank, idx in enumerate(retrieved.tolist(), start=1):
        if relevant[idx]:
            hits += 1
            ap += hits / rank
    return ap / n_relevant


def dcg_at_k(gains: torch.Tensor, k: int) -> float:
    """Discounted Cumulative Gain at rank K.

    Args:
        gains: 1-D tensor of relevance scores in ranked order.
        k: Cut-off rank.

    Returns:
        DCG@K value.
    """
    gains_k = gains[:k].float()
    positions = torch.arange(1, gains_k.size(0) + 1, dtype=torch.float32)
    discounts = torch.log2(positions + 1)
    return (gains_k / discounts).sum().item()


def ndcg_at_k(gains: torch.Tensor, k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank K.

    Args:
        gains: 1-D tensor of relevance scores in ranked order.
        k: Cut-off rank.

    Returns:
        NDCG@K in [0, 1]. Returns 0 if ideal DCG is 0.
    """
    actual = dcg_at_k(gains, k)
    ideal_gains, _ = torch.sort(gains, descending=True)
    ideal = dcg_at_k(ideal_gains, k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal
