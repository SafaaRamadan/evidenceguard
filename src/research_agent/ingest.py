"""Load text documents from a local directory."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    """One source file loaded into memory."""

    source: str  # file name only, e.g. "qwen.md"
    text: str


def load_documents(directory: Path | str) -> list[Document]:
    """Read every supported file in `directory` and return them as Documents."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    documents: list[Document] = []
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file: %s", file_path.name)
            continue

        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            logger.warning("Skipping file that is not valid UTF-8: %s", file_path.name)
            continue

        if not text:
            logger.warning("Skipping empty file: %s", file_path.name)
            continue

        documents.append(Document(source=file_path.name, text=text))

    if not documents:
        raise ValueError(f"No usable documents found in: {directory}")

    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = load_documents("data/raw")
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"- {doc.source}: {len(doc.text)} characters")