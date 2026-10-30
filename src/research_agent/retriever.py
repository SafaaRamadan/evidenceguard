import logging
from pathlib import Path
from research_agent.chunking import chunk_documents
from research_agent.embeddings import Embedder
from research_agent.index_store import IndexStore, SearchResult
from research_agent.ingest import load_documents

logger = logging.getLogger(__name__)


def build_index(data_dir: Path | str, embedder: Embedder, store: IndexStore) -> int:
    """Load, chunk, embed and store every document. Returns the chunk count."""
    chunks = chunk_documents(load_documents(data_dir))
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    store.rebuild(chunks, vectors)
    return len(chunks)


class Retriever:
    """Finds the chunks most relevant to a question."""
    def __init__(self, embedder: Embedder, store: IndexStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return the `top_k` best chunks for `question`, best first."""
        query_vector = self._embedder.embed_query(question)
        return self._store.search(query_vector, top_k=top_k)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    question = " ".join(sys.argv[1:]) or "Which license does Qwen3 use?"

    embedder = Embedder()
    store = IndexStore()
    build_index("data/raw", embedder, store)

    print(f"\nQ: {question}")
    for rank, hit in enumerate(Retriever(embedder, store).retrieve(question), start=1):
        print(f"  {rank}. {hit.score:.3f}  {hit.source} > {hit.section}")