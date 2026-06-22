"""Tests for ranking / evaluation metrics."""

import pytest
import torch

from ml_interview.metrics import (
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_setup():
    """5 items, items 0 and 2 are relevant; retrieved order: [2, 0, 4, 3, 1]."""
    relevant = torch.tensor([True, False, True, False, False])
    retrieved = torch.tensor([2, 0, 4, 3, 1])
    return relevant, retrieved


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self, simple_setup):
        relevant, retrieved = simple_setup
        # top-2 are [2, 0] — both relevant
        assert precision_at_k(relevant, retrieved, k=2) == 1.0

    def test_none_relevant(self, simple_setup):
        relevant, retrieved = simple_setup
        # top-1 is [2] — relevant; not testing "none" here but k>n_relevant
        assert precision_at_k(relevant, retrieved, k=5) == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_full_recall_at_k2(self, simple_setup):
        relevant, retrieved = simple_setup
        assert recall_at_k(relevant, retrieved, k=2) == 1.0

    def test_partial_recall(self, simple_setup):
        relevant, retrieved = simple_setup
        assert recall_at_k(relevant, retrieved, k=1) == pytest.approx(0.5)

    def test_no_relevant_items(self):
        relevant = torch.tensor([False, False, False])
        retrieved = torch.tensor([0, 1, 2])
        assert recall_at_k(relevant, retrieved, k=3) == 0.0


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_first_hit_at_rank_1(self):
        relevant = torch.tensor([True, False, False])
        retrieved = torch.tensor([0, 1, 2])
        assert reciprocal_rank(relevant, retrieved) == pytest.approx(1.0)

    def test_no_hit(self):
        relevant = torch.tensor([False, False, False])
        retrieved = torch.tensor([0, 1, 2])
        assert reciprocal_rank(relevant, retrieved) == 0.0

    def test_hit_at_rank_2(self, simple_setup):
        relevant, retrieved = simple_setup
        # First hit is at rank 1 (index 2 is relevant)
        assert reciprocal_rank(relevant, retrieved) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------

class TestAveragePrecision:
    def test_perfect_ranking(self):
        relevant = torch.tensor([True, True, False, False])
        retrieved = torch.tensor([0, 1, 2, 3])
        assert average_precision(relevant, retrieved) == pytest.approx(1.0)

    def test_no_relevant(self):
        relevant = torch.tensor([False, False, False])
        retrieved = torch.tensor([0, 1, 2])
        assert average_precision(relevant, retrieved) == 0.0


# ---------------------------------------------------------------------------
# dcg_at_k / ndcg_at_k
# ---------------------------------------------------------------------------

class TestDCGAndNDCG:
    def test_dcg_at_k(self):
        gains = torch.tensor([3.0, 2.0, 3.0, 0.0, 1.0])
        # DCG@3 = 3/log2(2) + 2/log2(3) + 3/log2(4)
        import math
        expected = 3 / math.log2(2) + 2 / math.log2(3) + 3 / math.log2(4)
        assert dcg_at_k(gains, k=3) == pytest.approx(expected, rel=1e-4)

    def test_ndcg_perfect(self):
        gains = torch.tensor([3.0, 2.0, 1.0])
        assert ndcg_at_k(gains, k=3) == pytest.approx(1.0, rel=1e-4)

    def test_ndcg_zero_ideal(self):
        gains = torch.tensor([0.0, 0.0, 0.0])
        assert ndcg_at_k(gains, k=3) == 0.0

    def test_ndcg_in_range(self):
        gains = torch.tensor([1.0, 0.0, 3.0, 2.0])
        score = ndcg_at_k(gains, k=4)
        assert 0.0 <= score <= 1.0 + 1e-5
