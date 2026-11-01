import pytest

from research_agent.generator import GenerationResult
from research_agent.index_store import SearchResult
from research_agent.pipeline import RagPipeline
from research_agent.prompting import ABSTAIN_MESSAGE


def make_result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        title="Title",
        section="Section",
        text=f"Text of {chunk_id}",
        source_url=None,
        date_accessed=None,
        score=0.5,
    )


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def retrieve(self, question: str, top_k: int = 3) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("Query must not be empty.")
        return self._results[:top_k]


class FakeGenerator:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_messages: list[dict[str, str]] = []

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 200):
        self.last_messages = messages
        return GenerationResult(
            text=self._text, prompt_tokens=100, new_tokens=10, seconds=1.0
        )


def make_pipeline(text: str, top_k: int = 3) -> tuple[RagPipeline, FakeGenerator]:
    generator = FakeGenerator(text)
    retriever = FakeRetriever([make_result("a"), make_result("b"), make_result("c")])
    return RagPipeline(retriever, generator, top_k=top_k), generator


def test_well_cited_answer_passes_validation() -> None:
    pipeline, _ = make_pipeline("It is open [E1].")

    answer = pipeline.answer("Is it open?")

    assert answer.text == "It is open [E1]."
    assert answer.citations.passed
    assert not answer.abstained
    assert list(answer.evidence) == ["E1", "E2", "E3"]
    assert answer.prompt_tokens == 100
    assert answer.new_tokens == 10


def test_abstention_is_recognized() -> None:
    pipeline, _ = make_pipeline(ABSTAIN_MESSAGE)

    assert pipeline.answer("Anything?").abstained


def test_invented_citation_is_flagged() -> None:
    pipeline, _ = make_pipeline("It is open [E9].")

    answer = pipeline.answer("Is it open?")

    assert not answer.citations.passed
    assert "invalid_citation" in answer.citations.issues


def test_retrieved_evidence_reaches_the_model_as_user_content() -> None:
    pipeline, generator = make_pipeline("It is open [E1].")

    pipeline.answer("Is it open?")

    assert [m["role"] for m in generator.last_messages] == ["system", "user"]
    assert "[E1] Text of a" in generator.last_messages[1]["content"]


def test_top_k_limits_the_evidence() -> None:
    pipeline, _ = make_pipeline("It is open [E1].", top_k=2)

    assert list(pipeline.answer("Is it open?").evidence) == ["E1", "E2"]


def test_empty_question_raises() -> None:
    pipeline, _ = make_pipeline("It is open [E1].")

    with pytest.raises(ValueError):
        pipeline.answer("   ")


def test_invalid_settings_raise() -> None:
    generator = FakeGenerator("x")
    retriever = FakeRetriever([make_result("a")])

    with pytest.raises(ValueError):
        RagPipeline(retriever, generator, top_k=0)
    with pytest.raises(ValueError):
        RagPipeline(retriever, generator, max_new_tokens=0)