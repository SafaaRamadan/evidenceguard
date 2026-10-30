import pytest

from research_agent.chunking import _split_words, chunk_document
from research_agent.ingest import Document

SAMPLE = """# Demo Model

## Overview

Demo is a small model.

## Limitations

It can be wrong.

## Official source

https://example.com/demo

## Date accessed

2026-01-01
"""


def test_chunks_follow_sections_and_ids() -> None:
    chunks = chunk_document(Document("demo.md", SAMPLE))

    assert [c.section for c in chunks] == ["Overview", "Limitations"]
    assert [c.chunk_id for c in chunks] == ["demo-01", "demo-02"]


def test_metadata_is_stored_not_embedded() -> None:
    chunks = chunk_document(Document("demo.md", SAMPLE))

    for chunk in chunks:
        assert chunk.source == "demo.md"
        assert chunk.source_url == "https://example.com/demo"
        assert chunk.date_accessed == "2026-01-01"
        assert "example.com" not in chunk.text
        assert "2026-01-01" not in chunk.text


def test_text_starts_with_title_and_section() -> None:
    chunks = chunk_document(Document("demo.md", SAMPLE))

    assert chunks[0].text.startswith("Demo Model > Overview\n")
    assert "Demo is a small model." in chunks[0].text


def test_text_before_first_heading_becomes_introduction() -> None:
    text = "# T\n\nIntro text\n\n## A\n\nBody text"
    chunks = chunk_document(Document("x.md", text))

    assert [c.section for c in chunks] == ["Introduction", "A"]
    assert chunks[0].source_url is None
    assert chunks[0].date_accessed is None


def test_short_text_is_not_split() -> None:
    assert _split_words("a b c", max_words=100, overlap_words=20) == ["a b c"]


def test_long_text_is_split_with_overlap() -> None:
    words = [f"w{i}" for i in range(250)]

    pieces = _split_words(" ".join(words), max_words=100, overlap_words=20)

    assert len(pieces) == 3
    assert all(len(p.split()) <= 100 for p in pieces)
    assert pieces[0].split()[-20:] == pieces[1].split()[:20]  # shared words
    assert pieces[-1].split()[-1] == "w249"  # nothing lost at the end


def test_overlap_not_smaller_than_max_raises() -> None:
    with pytest.raises(ValueError):
        chunk_document(Document("demo.md", SAMPLE), max_words=10, overlap_words=10)