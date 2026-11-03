"""A LangGraph agent: retrieve, generate, validate, and retry only when it helps.

The graph (see docs/architecture.md):

    START -> retrieve -> generate -> validate -> done                    -> END
                 ^           ^           |-> widen  (abstained, optional) -> retrieve
                 |           |           |-> repair (citation problems)   -> generate
                 +-----------+-----------+

Every route is bounded by `max_attempts` generations, so the loop always ends.
"""

import logging
import time
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from research_agent.config import Settings
from research_agent.embeddings import Embedder
from research_agent.generator import GenerationResult, Generator
from research_agent.index_store import IndexStore, SearchResult
from research_agent.pipeline import RagAnswer
from research_agent.prompting import ABSTAIN_MESSAGE, build_prompt
from research_agent.retriever import Retriever, build_index
from research_agent.verifier import CitationReport, check_citations

logger = logging.getLogger(__name__)

STATUS_ANSWERED = "answered"  # structural checks passed
STATUS_ABSTAINED = "abstained"  # the model said the evidence is insufficient
STATUS_UNVERIFIED = "unverified"  # an answer exists but its citations failed checks

FEEDBACK = {
    "empty_answer": "Your answer was empty. Answer in one sentence and cite a passage like [E1].",
    "no_content": (
        "Your answer contained only citation tags. Write one complete sentence that "
        "answers the question, then cite the passage, like: "
        "Model X is released under the MIT license [E1]."
    ),
    "header_only": (
        "Your answer only repeated a passage title. Write one complete sentence that "
        "answers the question, then cite the passage, like: "
        "Model X is released under the MIT license [E1]."
    ),
    "no_citations": (
        "Your answer had no valid citation. Cite passages only as [E1], [E2], ... "
        "at the end of the sentence they support. Do not put titles or names in brackets."
    ),
    "invalid_citation": "You cited passage IDs that do not exist. Use only: {valid_ids}.",
    "uncited_sentence": "Every sentence must end with a citation such as [E1]. Rewrite it so each sentence is cited.",
}


@dataclass(frozen=True)
class AgentAnswer(RagAnswer):
    """A RagAnswer plus how the agent got there."""

    attempts: int = 1
    status: str = STATUS_ANSWERED
    trace: tuple[str, ...] = ()


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    widened: bool
    hits: list[SearchResult]
    evidence: dict[str, SearchResult]
    messages: list[dict[str, str]]
    generation: GenerationResult
    report: CitationReport
    attempts: int
    retrieval_seconds: float
    generation_seconds: float
    trace: list[str]


class EvidenceAgent:
    """Answers questions from the indexed documents, with bounded retries."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 3,
        widen_k: int = 5,
        max_attempts: int = 3,
        max_new_tokens: int = 150,
        retry_on_abstain: bool = False,
        min_retry_score: float = 0.35,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if widen_k <= top_k:
            raise ValueError("widen_k must be larger than top_k.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._widen_k = widen_k
        self._max_attempts = max_attempts
        self._max_new_tokens = max_new_tokens
        self._retry_on_abstain = retry_on_abstain
        self._min_retry_score = min_retry_score
        self._graph = self._build_graph()

    # ---- graph nodes: each returns only the state keys it changes ----

    def _retrieve(self, state: AgentState) -> dict:
        start = time.perf_counter()
        hits = self._retriever.retrieve(state["question"], top_k=state["top_k"])
        elapsed = time.perf_counter() - start
        prompt = build_prompt(state["question"], hits)
        return {
            "hits": hits,
            "evidence": prompt.evidence,
            "messages": prompt.messages,
            "retrieval_seconds": state.get("retrieval_seconds", 0.0) + elapsed,
            "trace": [*state.get("trace", []), f"retrieved top {state['top_k']}"],
        }

    def _generate(self, state: AgentState) -> dict:
        result = self._generator.generate(state["messages"], max_new_tokens=self._max_new_tokens)
        return {
            "generation": result,
            "attempts": state.get("attempts", 0) + 1,
            "generation_seconds": state.get("generation_seconds", 0.0) + result.seconds,
        }

    def _validate(self, state: AgentState) -> dict:
        report = check_citations(state["generation"].text, set(state["evidence"]))
        outcome = "abstained" if report.abstained else (", ".join(report.issues) or "ok")
        return {
            "report": report,
            "trace": [*state["trace"], f"attempt {state['attempts']}: {outcome}"],
        }

    def _route(self, state: AgentState) -> str:
        report = state["report"]
        if state["attempts"] >= self._max_attempts:
            return "done"
        if report.abstained:
            looks_relevant = state["hits"][0].score >= self._min_retry_score
            if self._retry_on_abstain and not state.get("widened") and looks_relevant:
                return "widen"
            return "done"
        if report.passed:
            return "done"
        return "repair"

    def _widen(self, state: AgentState) -> dict:
        return {
            "top_k": self._widen_k,
            "widened": True,
            "trace": [*state["trace"], f"widening retrieval to top {self._widen_k}"],
        }

    def _repair(self, state: AgentState) -> dict:
        report = state["report"]
        valid_ids = ", ".join(f"[{evidence_id}]" for evidence_id in state["evidence"])
        notes = [
            FEEDBACK.get(issue, f"Fix this problem: {issue}.").format(valid_ids=valid_ids)
            for issue in report.issues
        ]
        feedback = " ".join(notes) + f" If the evidence does not contain the answer, reply exactly: {ABSTAIN_MESSAGE}"
        messages = [
            *state["messages"],
            {"role": "assistant", "content": state["generation"].text},
            {"role": "user", "content": feedback},
        ]
        return {"messages": messages, "trace": [*state["trace"], "asking the model to repair its answer"]}

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)
        graph.add_node("widen", self._widen)
        graph.add_node("repair", self._repair)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")
        graph.add_conditional_edges(
            "validate", self._route, {"done": END, "widen": "widen", "repair": "repair"}
        )
        graph.add_edge("widen", "retrieve")
        graph.add_edge("repair", "generate")
        return graph.compile()

    def diagram(self) -> str:
        """The graph as Mermaid text, for documentation."""
        return self._graph.get_graph().draw_mermaid()

    def answer(self, question: str) -> AgentAnswer:
        """Run the graph for one question."""
        if not question.strip():
            raise ValueError("Question must not be empty.")
        final = self._graph.invoke(
            {"question": question.strip(), "top_k": self._top_k, "attempts": 0, "trace": []}
        )
        report = final["report"]
        generation = final["generation"]
        if report.abstained:
            status = STATUS_ABSTAINED
        elif report.passed:
            status = STATUS_ANSWERED
        else:
            status = STATUS_UNVERIFIED
        return AgentAnswer(
            question=question.strip(),
            text=generation.text,
            abstained=report.abstained,
            evidence=final["evidence"],
            citations=report,
            prompt_tokens=generation.prompt_tokens,
            new_tokens=generation.new_tokens,
            retrieval_seconds=final["retrieval_seconds"],
            generation_seconds=final["generation_seconds"],
            attempts=final["attempts"],
            status=status,
            trace=tuple(final["trace"]),
        )


def build_agent(settings: Settings | None = None) -> EvidenceAgent:
    """Load the models, index the documents, and return a ready agent."""
    settings = settings or Settings.from_env()
    embedder = Embedder()
    store = IndexStore(settings.db_path)
    build_index(settings.data_dir, embedder, store)
    generator = Generator(model_name=settings.llm_name, dtype_name=settings.llm_dtype)
    return EvidenceAgent(
        Retriever(embedder, store),
        generator,
        top_k=settings.top_k,
        widen_k=settings.widen_k,
        max_attempts=settings.max_attempts,
        max_new_tokens=settings.max_new_tokens,
        retry_on_abstain=settings.retry_on_abstain,
        min_retry_score=settings.min_retry_score,
    )