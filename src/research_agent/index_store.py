import logging
from dataclasses import dataclass
from pathlib import Path
import chromadb
from chromadb.config import Settings
from research_agent.chunking import Chunk

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("chroma_db")
COLLECTION_NAME = "evidenceguard_chunks"
COLLECTION_CONFIG = {"hnsw": {"space": "cosine"}}


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source: str
    title: str
    section: str
    text: str
    source_url: str | None
    date_accessed: str | None
    score: float  # cosine similarity: higher means more similar


class IndexStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration=COLLECTION_CONFIG,
        )

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def rebuild(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace everything stored with these chunks and their vectors."""
        if not chunks:
            raise ValueError("No chunks to index.")
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors; "
                "they must match one to one."
            )

        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.create_collection(
            name=COLLECTION_NAME,
            configuration=COLLECTION_CONFIG,
        )
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_to_metadata(chunk) for chunk in chunks],
        )
        logger.info("Indexed %d chunks", len(chunks))

    def search(self, query_vector: list[float], top_k: int = 3) -> list[SearchResult]:
        """Return the `top_k` chunks closest to the query vector."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        stored = self.count()
        if stored == 0:
            raise RuntimeError("The index is empty. Call rebuild() first.")

        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, stored),
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            SearchResult(
                chunk_id=chunk_id,
                source=meta["source"],
                title=meta["title"],
                section=meta["section"],
                text=document,
                source_url=meta.get("source_url"),
                date_accessed=meta.get("date_accessed"),
                score=1.0 - distance,
            )
            for chunk_id, document, meta, distance in zip(
                ids, documents, metadatas, distances
            )
        ]


def _to_metadata(chunk: Chunk) -> dict[str, str]:
    """Build ChromaDB metadata, leaving out values that are None."""
    metadata = {
        "source": chunk.source,
        "title": chunk.title,
        "section": chunk.section,
    }
    if chunk.source_url is not None:
        metadata["source_url"] = chunk.source_url
    if chunk.date_accessed is not None:
        metadata["date_accessed"] = chunk.date_accessed
    return metadata


if __name__ == "__main__":
    from research_agent.chunking import chunk_documents
    from research_agent.embeddings import Embedder
    from research_agent.ingest import load_documents

    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    chunks = chunk_documents(load_documents("data/raw"))
    embedder = Embedder()
    store = IndexStore()
    store.rebuild(chunks, embedder.embed_texts([chunk.text for chunk in chunks]))
    print(f"Stored {store.count()} chunks")

    questions = [
        "Which license does Qwen3 use?",
        "What are the limitations of Mistral?",
        "Which model can understand images?",
    ]
    for question in questions:
        print(f"\nQ: {question}")
        for hit in store.search(embedder.embed_query(question), top_k=3):
            print(f"  {hit.score:.3f}  {hit.chunk_id:<12} {hit.section}")