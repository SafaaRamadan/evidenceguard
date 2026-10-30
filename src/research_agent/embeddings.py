import logging
import time

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        start = time.perf_counter()
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        logger.info("Loaded %s in %.1f s", model_name, time.perf_counter() - start)

    @property
    def max_tokens(self) -> int:
        """Longest input (in tokens) the model reads; the rest is cut off."""
        return self._model.max_seq_length

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Number of tokens in each text, to detect silent truncation."""
        return [len(self._model.tokenizer.encode(text)) for text in texts]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts at once. Vectors have length 1 (normalized)."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        return self.embed_texts([query])[0]


if __name__ == "__main__":
    from research_agent.chunking import chunk_documents
    from research_agent.ingest import load_documents

    logging.basicConfig(level=logging.INFO)
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    chunks = chunk_documents(load_documents("data/raw"))
    texts = [chunk.text for chunk in chunks]

    embedder = Embedder()

    start = time.perf_counter()
    vectors = embedder.embed_texts(texts)
    elapsed = time.perf_counter() - start

    token_counts = embedder.count_tokens(texts)
    print(f"Chunks embedded: {len(vectors)} in {elapsed:.2f} s")
    print(f"Vector size: {len(vectors[0])}")
    print(f"Vector length (should be ~1.0): {sum(x * x for x in vectors[0]) ** 0.5:.3f}")
    print(f"Longest chunk: {max(token_counts)} tokens (model limit: {embedder.max_tokens})")