from pathlib import Path

import pytest

from research_agent.ingest import load_documents


def test_loads_supported_files_only(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("Alpha document", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Beta document", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("not supported yet", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [doc.source for doc in documents] == ["a.md", "b.txt"]
    assert documents[0].text == "Alpha document"


def test_skips_empty_files(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   \n\n", encoding="utf-8")
    (tmp_path / "real.md").write_text("Some content", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [doc.source for doc in documents] == ["real.md"]


def test_skips_file_with_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe\xfa")
    (tmp_path / "good.txt").write_text("Good content", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [doc.source for doc in documents] == ["good.txt"]


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does_not_exist")


def test_directory_without_usable_documents_raises(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_documents(tmp_path)