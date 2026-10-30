import logging
import re
from dataclasses import dataclass
from pathlib import Path

from research_agent.ingest import Document

logger = logging.getLogger(__name__)

MAX_WORDS = 120
OVERLAP_WORDS = 20
SOURCE_SECTION = "official source"
DATE_SECTION = "date accessed"
METADATA_SECTIONS = {SOURCE_SECTION, DATE_SECTION}
URL_PATTERN = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class Chunk:
    #One retrievable piece of a document, with its metadata.
    chunk_id: str  # e.g. "qwen-02"
    source: str  # file name, e.g. "qwen.md"
    title: str  # document title, e.g. "Qwen Language Models"
    section: str  # section heading, e.g. "Limitations"
    text: str  # "<title> > <section>\n<content>"
    source_url: str | None  # official URL from the document, if present
    date_accessed: str | None # date accessed from the document, if present


def _parse_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    title = "Untitled"
    headings: list[str] = []
    bodies: list[list[str]] = []

    for line in text.splitlines():
        if line.startswith("## "):
            headings.append(line[3:].strip())
            bodies.append([])
        elif line.startswith("# "):
            title = line[2:].strip()
        elif line.strip():
            if not headings:  # text before the first "##" heading
                headings.append("Introduction")
                bodies.append([])
            bodies[-1].append(line)

    sections = [(h, "\n".join(b).strip()) for h, b in zip(headings, bodies)]
    return title, sections


def _split_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    #Split text into windows of at most `max_words`, overlapping slightly.
    words = text.split()
    if len(words) <= max_words:
        return [text]

    step = max_words - overlap_words
    pieces: list[str] = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return pieces


def chunk_document(
    document: Document,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    #Split one document into chunks, one or more per section.
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    title, sections = _parse_sections(document.text)

    source_url = None
    date_accessed = None
    for heading, body in sections:
        if heading.lower() == SOURCE_SECTION:
            match = URL_PATTERN.search(body)
            source_url = match.group(0) if match else None
        elif heading.lower() == DATE_SECTION:
            date_accessed = body or None

    stem = Path(document.source).stem
    chunks: list[Chunk] = []
    for heading, body in sections:
        if heading.lower() in METADATA_SECTIONS or not body:
            continue
        for piece in _split_words(body, max_words, overlap_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}-{len(chunks) + 1:02d}",
                    source=document.source,
                    title=title,
                    section=heading,
                    text=f"{title} > {heading}\n{piece}",
                    source_url=source_url,
                    date_accessed=date_accessed,
                )
            )
    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk every document and return one flat list."""
    chunks = [chunk for doc in documents for chunk in chunk_document(doc)]
    logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
    return chunks


if __name__ == "__main__":
    from research_agent.ingest import load_documents

    logging.basicConfig(level=logging.INFO)
    all_chunks = chunk_documents(load_documents("data/raw"))
    for chunk in all_chunks:
        n_words = len(chunk.text.split())
        print(f"{chunk.chunk_id:<12} {chunk.section:<14} {n_words:>3} words")