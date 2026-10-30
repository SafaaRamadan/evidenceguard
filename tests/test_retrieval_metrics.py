import pytest

from research_agent.retrieval_metrics import hit_at_k, recall_at_k, reciprocal_rank

A, B, C = ("a.md", "X"), ("b.md", "Y"), ("c.md", "Z")
MISSING = ("z.md", "Q")
RETRIEVED = [A, B, C]  # ranks 1, 2, 3


def test_hit_at_k() -> None:
    assert hit_at_k(RETRIEVED, [B], k=1) == 0.0
    assert hit_at_k(RETRIEVED, [B], k=2) == 1.0
    assert hit_at_k(RETRIEVED, [MISSING], k=3) == 0.0


def test_recall_at_k() -> None:
    gold = [A, C]

    assert recall_at_k(RETRIEVED, gold, k=1) == 0.5
    assert recall_at_k(RETRIEVED, gold, k=2) == 0.5
    assert recall_at_k(RETRIEVED, gold, k=3) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RETRIEVED, [A]) == 1.0
    assert reciprocal_rank(RETRIEVED, [B]) == 0.5
    assert reciprocal_rank(RETRIEVED, [C]) == pytest.approx(1 / 3)
    assert reciprocal_rank(RETRIEVED, [C, B]) == 0.5  # the first gold found wins
    assert reciprocal_rank(RETRIEVED, [MISSING]) == 0.0


def test_duplicate_sections_do_not_inflate_recall() -> None:
    assert recall_at_k([A, A, B], [A], k=3) == 1.0


def test_metrics_reject_empty_gold() -> None:
    with pytest.raises(ValueError):
        hit_at_k(RETRIEVED, [], k=3)
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, [], k=3)
    with pytest.raises(ValueError):
        reciprocal_rank(RETRIEVED, [])


def test_metrics_reject_bad_k() -> None:
    with pytest.raises(ValueError):
        hit_at_k(RETRIEVED, [A], k=0)
    with pytest.raises(ValueError):
        recall_at_k(RETRIEVED, [A], k=0)