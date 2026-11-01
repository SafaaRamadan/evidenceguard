# A straight-line RAG pipeline: retrieve, build the prompt, generate, validate.
import time
from dataclasses import dataclass

from research_agent.generator import Generator
from research_agent.index_store import SearchResult
from research_agent.prompting import build_prompt
from research_agent.retriever import Retriever
from research_agent.verifier import CitationReport, check_citations


@dataclass(frozen=True)
class RagAnswer:
    """An answer together with the evidence and measurements behind it."""
    question: str
    text: str
    abstained: bool
    evidence: dict[str, SearchResult]  # "E1" -> chunk shown to the model
    citations: CitationReport
    prompt_tokens: int
    new_tokens: int
    retrieval_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 3,
        max_new_tokens: int = 150,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._max_new_tokens = max_new_tokens

    def answer(self, question: str) -> RagAnswer:
        """Run the full pipeline for one question."""
        start = time.perf_counter()
        hits = self._retriever.retrieve(question, top_k=self._top_k)
        retrieval_seconds = time.perf_counter() - start

        prompt = build_prompt(question, hits)
        result = self._generator.generate(
            prompt.messages, max_new_tokens=self._max_new_tokens
        )
        report = check_citations(result.text, set(prompt.evidence))

        return RagAnswer(
            question=question.strip(),
            text=result.text,
            abstained=report.abstained,
            evidence=prompt.evidence,
            citations=report,
            prompt_tokens=result.prompt_tokens,
            new_tokens=result.new_tokens,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=result.seconds,
        )