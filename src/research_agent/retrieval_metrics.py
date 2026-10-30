Pair = tuple[str, str]  # (source file name, section heading)


def _validate(gold: list[Pair], k: int | None = None) -> None:
    if not gold:
        raise ValueError(
            "gold must not be empty: retrieval metrics are undefined for "
            "questions that have no answer in the corpus."
        )
    if k is not None and k < 1:
        raise ValueError("k must be at least 1.")


def hit_at_k(retrieved: list[Pair], gold: list[Pair], k: int) -> float:
    """1.0 if at least one gold section is in the top k results, else 0.0."""
    _validate(gold, k)
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def recall_at_k(retrieved: list[Pair], gold: list[Pair], k: int) -> float:
    """Fraction of the gold sections that appear in the top k results."""
    _validate(gold, k)
    found = set(retrieved[:k]) & set(gold)
    return len(found) / len(set(gold))


def reciprocal_rank(retrieved: list[Pair], gold: list[Pair]) -> float:
    """1 / rank of the first gold section found; 0.0 if none was retrieved."""
    _validate(gold)
    for rank, pair in enumerate(retrieved, start=1):
        if pair in gold:
            return 1.0 / rank
    return 0.0