import math
import pytest
from research_agent.embeddings import Embedder

pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_vectors_have_same_size_and_unit_length(embedder: Embedder) -> None:
    vectors = embedder.embed_texts(["first text", "a second, longer text"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0
    for vector in vectors:
        assert math.isclose(math.sqrt(dot(vector, vector)), 1.0, rel_tol=1e-3)


def test_empty_list_returns_empty_list(embedder: Embedder) -> None:
    assert embedder.embed_texts([]) == []


def test_empty_query_raises(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.embed_query("   ")


def test_query_embedding_matches_batch_embedding(embedder: Embedder) -> None:
    single = embedder.embed_query("What is RAG?")
    batch = embedder.embed_texts(["What is RAG?"])[0]

    assert single == pytest.approx(batch, abs=1e-5)


def test_related_text_scores_higher_than_unrelated(embedder: Embedder) -> None:
    question = embedder.embed_query("Which license does Qwen3 use?")
    related, unrelated = embedder.embed_texts(
        [
            "The Qwen3 open-weight models are licensed under Apache 2.0.",
            "Mix the rice, pasta and lentils, then top with fried onions.",
        ]
    )

    assert dot(question, related) > dot(question, unrelated) + 0.3


def test_count_tokens_grows_with_text_length(embedder: Embedder) -> None:
    short, long = embedder.count_tokens(
        ["short", "a much longer sentence with many more words in it"]
    )

    assert short < long