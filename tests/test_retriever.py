from pathlib import Path
import pytest
from research_agent.index_store import IndexStore
from research_agent.retriever import Retriever, build_index


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def write_corpus(directory: Path) -> None:
    (directory / "alpha.md").write_text(
        "# Alpha Model\n\n## Overview\n\nAlpha is fast.\n\n## Licensing\n\nAlpha is open.",
        encoding="utf-8",
    )
    (directory / "beta.md").write_text(
        "# Beta Model\n\n## Overview\n\nBeta is small.",
        encoding="utf-8",
    )


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_corpus(corpus)
    embedder = FakeEmbedder()
    store = IndexStore(tmp_path / "db")
    build_index(corpus, embedder, store)
    return Retriever(embedder, store)


def test_build_index_returns_chunk_count(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_corpus(corpus)
    store = IndexStore(tmp_path / "db")

    count = build_index(corpus, FakeEmbedder(), store)

    assert count == 3
    assert store.count() == 3


def test_retrieve_puts_matching_source_first(retriever: Retriever) -> None:
    hits = retriever.retrieve("tell me about beta", top_k=1)

    assert len(hits) == 1
    assert hits[0].source == "beta.md"


def test_retrieve_respects_top_k(retriever: Retriever) -> None:
    hits = retriever.retrieve("tell me about alpha", top_k=2)

    assert len(hits) == 2
    assert {hit.source for hit in hits} == {"alpha.md"}


def test_empty_question_raises(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("   ")