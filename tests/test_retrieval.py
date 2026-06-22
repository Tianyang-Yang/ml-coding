"""Tests for retrieval utilities."""

import torch
import pytest

from ml_interview.retrieval import (
    cosine_similarity_matrix,
    dot_product_similarity_matrix,
    top_k_retrieval,
)


class TestCosineSimilarityMatrix:
    def test_shape(self):
        a = torch.randn(3, 16)
        b = torch.randn(5, 16)
        sim = cosine_similarity_matrix(a, b)
        assert sim.shape == (3, 5)

    def test_identical_vectors_similarity_one(self):
        a = torch.randn(4, 8)
        sim = cosine_similarity_matrix(a, a)
        assert torch.allclose(sim.diag(), torch.ones(4), atol=1e-5)

    def test_values_in_range(self):
        a = torch.randn(10, 32)
        b = torch.randn(10, 32)
        sim = cosine_similarity_matrix(a, b)
        assert (sim >= -1.0 - 1e-5).all() and (sim <= 1.0 + 1e-5).all()


class TestDotProductSimilarityMatrix:
    def test_shape(self):
        a = torch.randn(4, 16)
        b = torch.randn(6, 16)
        sim = dot_product_similarity_matrix(a, b)
        assert sim.shape == (4, 6)

    def test_correct_values(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        sim = dot_product_similarity_matrix(a, b)
        expected = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        assert torch.allclose(sim, expected)


class TestTopKRetrieval:
    def test_output_shapes(self):
        q = torch.randn(4, 16)
        c = torch.randn(20, 16)
        scores, indices = top_k_retrieval(q, c, k=5)
        assert scores.shape == (4, 5)
        assert indices.shape == (4, 5)

    def test_indices_in_valid_range(self):
        q = torch.randn(3, 8)
        c = torch.randn(10, 8)
        _, indices = top_k_retrieval(q, c, k=3)
        assert (indices >= 0).all() and (indices < 10).all()

    def test_dot_product_mode(self):
        q = torch.randn(2, 16)
        c = torch.randn(8, 16)
        scores, indices = top_k_retrieval(q, c, k=3, normalize=False)
        assert scores.shape == (2, 3)

    def test_scores_descending(self):
        """Top-k scores should be in descending order."""
        q = torch.randn(5, 32)
        c = torch.randn(50, 32)
        scores, _ = top_k_retrieval(q, c, k=10)
        diffs = scores[:, :-1] - scores[:, 1:]
        assert (diffs >= -1e-5).all()
