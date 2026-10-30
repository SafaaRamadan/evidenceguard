from pathlib import Path

import pytest

from research_agent.chunking import Chunk
from research_agent.index_store import IndexStore

A, B, C = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]


def make_chunk(
    chunk_id: str,
    source_url: str | None = "https://example.com",
    date_accessed: str | None = "2026-01-01",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        title="Title",
        section="Section",
        text=f"Text of {chunk_id}",
        source_url=source_url,
        date_accessed=date_accessed,
    )


def test_search_returns_closest_first(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")
    store.rebuild([make_chunk("a"), make_chunk("b"), make_chunk("c")], [A, B, C])

    results = store.search([1.0, 0.0, 0.0], top_k=3)

    assert [r.chunk_id for r in results] == ["a", "c", "b"]
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert results[2].score == pytest.approx(0.0, abs=1e-3)


def test_metadata_and_text_round_trip(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")
    store.rebuild([make_chunk("a")], [A])

    hit = store.search(A, top_k=1)[0]

    assert hit.source == "a.md"
    assert hit.section == "Section"
    assert hit.text == "Text of a"
    assert hit.source_url == "https://example.com"
    assert hit.date_accessed == "2026-01-01"


def test_missing_metadata_comes_back_as_none(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")
    store.rebuild([make_chunk("a", source_url=None, date_accessed=None)], [A])

    hit = store.search(A, top_k=1)[0]

    assert hit.source_url is None
    assert hit.date_accessed is None


def test_rebuild_replaces_old_chunks(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")
    store.rebuild([make_chunk("a"), make_chunk("b")], [A, B])
    store.rebuild([make_chunk("c")], [C])

    assert store.count() == 1
    assert store.search(C, top_k=5)[0].chunk_id == "c"


def test_data_persists_between_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "db"
    IndexStore(db_path).rebuild([make_chunk("a"), make_chunk("b")], [A, B])

    assert IndexStore(db_path).count() == 2


def test_top_k_larger_than_stored_returns_all(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")
    store.rebuild([make_chunk("a"), make_chunk("b")], [A, B])

    assert len(store.search(A, top_k=10)) == 2


def test_mismatched_lengths_raise(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")

    with pytest.raises(ValueError):
        store.rebuild([make_chunk("a"), make_chunk("b")], [A])


def test_empty_chunk_list_raises(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")

    with pytest.raises(ValueError):
        store.rebuild([], [])


def test_search_on_empty_index_raises(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")

    with pytest.raises(RuntimeError):
        store.search(A, top_k=1)


def test_top_k_below_one_raises(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "db")
    store.rebuild([make_chunk("a")], [A])

    with pytest.raises(ValueError):
        store.search(A, top_k=0)