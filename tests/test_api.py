import pytest
from fastapi.testclient import TestClient

from research_agent.agent import AgentAnswer
from research_agent.api import MAX_QUESTION_CHARS, create_app
from research_agent.index_store import SearchResult
from research_agent.verifier import check_citations


def make_hit(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        title="Title",
        section="Licensing",
        text=f"Text of {chunk_id}",
        source_url="https://example.com/doc",
        date_accessed="2026-01-01",
        score=0.81234,
    )


class FakeAgent:
    def __init__(self, text: str = "It is open [E1].", fail: bool = False) -> None:
        self._text = text
        self._fail = fail

    def answer(self, question: str) -> AgentAnswer:
        if self._fail:
            raise RuntimeError("secret internal detail")
        evidence = {"E1": make_hit("a"), "E2": make_hit("b")}
        report = check_citations(self._text, set(evidence))
        return AgentAnswer(
            question=question,
            text=self._text,
            abstained=report.abstained,
            evidence=evidence,
            citations=report,
            prompt_tokens=100,
            new_tokens=10,
            retrieval_seconds=0.02,
            generation_seconds=2.0,
            attempts=2,
            status="answered" if report.passed else "unverified",
            trace=("retrieved top 3",),
        )


def client_for(agent: FakeAgent | None) -> TestClient:
    return TestClient(create_app(agent), raise_server_exceptions=False)


def test_health_reports_readiness() -> None:
    assert client_for(FakeAgent()).get("/health").json() == {"status": "ok", "ready": True}
    assert client_for(None).get("/health").json() == {"status": "ok", "ready": False}


def test_ask_returns_a_structured_answer() -> None:
    response = client_for(FakeAgent()).post("/ask", json={"question": "  Is it open?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Is it open?"
    assert body["status"] == "answered"
    assert body["answer"] == "It is open [E1]."
    assert body["citations_valid"] is True
    assert body["issues"] == []
    assert body["attempts"] == 2
    assert body["latency_seconds"] >= 0
    assert "not that the answer is factually correct" in body["note"]
    first, second = body["evidence"]
    assert first["evidence_id"] == "E1" and first["cited"] is True
    assert second["evidence_id"] == "E2" and second["cited"] is False
    assert first["source"] == "a.md"
    assert first["score"] == 0.812


def test_invalid_citations_are_reported_not_hidden() -> None:
    response = client_for(FakeAgent(text="It is open [E9].")).post(
        "/ask", json={"question": "Is it open?"}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "unverified"
    assert body["citations_valid"] is False
    assert body["issues"] == ["invalid_citation"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
        {"question": 123},
        {"wrong_field": "Is it open?"},
    ],
)
def test_bad_input_is_rejected_with_422(payload: dict) -> None:
    assert client_for(FakeAgent()).post("/ask", json=payload).status_code == 422


def test_unexpected_errors_do_not_leak_details() -> None:
    response = client_for(FakeAgent(fail=True)).post("/ask", json={"question": "Is it open?"})

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_ask_before_startup_returns_503() -> None:
    response = client_for(None).post("/ask", json={"question": "Is it open?"})

    assert response.status_code == 503