import pytest

from research_agent.agent import EvidenceAgent
from research_agent.generator import GenerationResult
from research_agent.index_store import SearchResult
from research_agent.prompting import ABSTAIN_MESSAGE

GOOD = "It is open [E1]."
CITATION_ONLY = "[E1]"
NO_CITATION = "It is open."


def make_hit(chunk_id: str, score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        title="Title",
        section="Section",
        text=f"Text of {chunk_id}",
        source_url=None,
        date_accessed=None,
        score=score,
    )


class FakeRetriever:
    def __init__(self, top_score: float = 0.8) -> None:
        self.calls: list[int] = []
        self._hits = [make_hit(name, top_score if name == "a" else 0.3) for name in "abcdefg"]

    def retrieve(self, question: str, top_k: int = 3) -> list[SearchResult]:
        self.calls.append(top_k)
        return self._hits[:top_k]


class FakeGenerator:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 200):
        self.calls.append(messages)
        text = self._texts[min(len(self.calls) - 1, len(self._texts) - 1)]
        return GenerationResult(text=text, prompt_tokens=100, new_tokens=10, seconds=1.0)


def make_agent(texts: list[str], top_score: float = 0.8, **options):
    retriever, generator = FakeRetriever(top_score), FakeGenerator(texts)
    return EvidenceAgent(retriever, generator, **options), retriever, generator


def test_good_answer_needs_one_attempt() -> None:
    agent, _, generator = make_agent([GOOD])

    answer = agent.answer("Is it open?")

    assert answer.status == "answered"
    assert answer.attempts == 1
    assert answer.text == GOOD
    assert len(generator.calls) == 1
    assert answer.generation_seconds == pytest.approx(1.0)


def test_citation_only_answer_is_repaired() -> None:
    agent, _, generator = make_agent([CITATION_ONLY, GOOD])

    answer = agent.answer("Is it open?")

    assert answer.status == "answered"
    assert answer.attempts == 2
    assert answer.text == GOOD
    second_call = generator.calls[1]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    assert second_call[2]["content"] == CITATION_ONLY
    assert "only citation tags" in second_call[3]["content"]
    assert answer.generation_seconds == pytest.approx(2.0)


def test_copied_header_is_repaired() -> None:
    agent, _, generator = make_agent(["[E1] Title > Section", GOOD])

    answer = agent.answer("Is it open?")

    assert answer.status == "answered"
    assert answer.attempts == 2
    assert "only repeated a passage title" in generator.calls[1][3]["content"]


def test_invalid_id_feedback_lists_the_valid_ids() -> None:
    agent, _, generator = make_agent(["It is open [E9].", GOOD])

    agent.answer("Is it open?")

    assert "[E1], [E2], [E3]" in generator.calls[1][3]["content"]


def test_answer_that_never_passes_is_marked_unverified() -> None:
    agent, _, generator = make_agent([NO_CITATION], max_attempts=3)

    answer = agent.answer("Is it open?")

    assert answer.status == "unverified"
    assert answer.attempts == 3
    assert len(generator.calls) == 3
    assert "no_citations" in answer.citations.issues
    assert not answer.abstained


def test_max_attempts_one_disables_repair() -> None:
    agent, _, generator = make_agent([CITATION_ONLY, GOOD], max_attempts=1)

    answer = agent.answer("Is it open?")

    assert answer.status == "unverified"
    assert len(generator.calls) == 1


def test_abstention_is_final_by_default() -> None:
    agent, retriever, generator = make_agent([ABSTAIN_MESSAGE])

    answer = agent.answer("Is it open?")

    assert answer.status == "abstained"
    assert answer.abstained
    assert answer.attempts == 1
    assert retriever.calls == [3]


def test_retry_on_abstain_widens_retrieval_once() -> None:
    agent, retriever, _ = make_agent([ABSTAIN_MESSAGE, GOOD], retry_on_abstain=True)

    answer = agent.answer("Is it open?")

    assert retriever.calls == [3, 5]
    assert answer.status == "answered"
    assert answer.attempts == 2
    assert len(answer.evidence) == 5


def test_widening_happens_only_once() -> None:
    agent, retriever, _ = make_agent([ABSTAIN_MESSAGE], retry_on_abstain=True)

    answer = agent.answer("Is it open?")

    assert retriever.calls == [3, 5]
    assert answer.status == "abstained"
    assert answer.attempts == 2


def test_weak_evidence_is_not_retried() -> None:
    agent, retriever, _ = make_agent(
        [ABSTAIN_MESSAGE], top_score=0.2, retry_on_abstain=True, min_retry_score=0.35
    )

    answer = agent.answer("Is it open?")

    assert retriever.calls == [3]
    assert answer.attempts == 1


def test_trace_describes_what_happened() -> None:
    agent, _, _ = make_agent([CITATION_ONLY, GOOD])

    trace = agent.answer("Is it open?").trace

    assert trace == (
        "retrieved top 3",
        "attempt 1: no_content",
        "asking the model to repair its answer",
        "attempt 2: ok",
    )


def test_diagram_mentions_every_node() -> None:
    agent, _, _ = make_agent([GOOD])

    diagram = agent.diagram()

    for node in ("retrieve", "generate", "validate", "widen", "repair"):
        assert node in diagram


def test_empty_question_raises() -> None:
    agent, _, _ = make_agent([GOOD])

    with pytest.raises(ValueError):
        agent.answer("   ")


@pytest.mark.parametrize(
    "options",
    [{"top_k": 0}, {"widen_k": 3}, {"max_attempts": 0}, {"max_new_tokens": 0}],
)
def test_invalid_settings_raise(options: dict) -> None:
    with pytest.raises(ValueError):
        EvidenceAgent(FakeRetriever(), FakeGenerator([GOOD]), **options)