import pytest

from research_agent import cli
from research_agent.agent import AgentAnswer
from research_agent.index_store import SearchResult
from research_agent.verifier import check_citations


def make_hit(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        title="Title",
        section="Licensing",
        text=f"Text of {chunk_id}",
        source_url=None,
        date_accessed=None,
        score=0.8,
    )


class FakeAgent:
    def answer(self, question: str) -> AgentAnswer:
        evidence = {"E1": make_hit("a"), "E2": make_hit("b")}
        text = "It is open [E1]."
        return AgentAnswer(
            question=question,
            text=text,
            abstained=False,
            evidence=evidence,
            citations=check_citations(text, set(evidence)),
            prompt_tokens=100,
            new_tokens=10,
            retrieval_seconds=0.02,
            generation_seconds=2.0,
            attempts=1,
            status="answered",
        )


def test_cli_prints_answer_status_and_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "build_agent", lambda: FakeAgent())

    code = cli.main(["Is", "it", "open?"])

    output = capsys.readouterr().out
    assert code == 0
    assert "Q: Is it open?" in output
    assert "Status: answered" in output
    assert "It is open [E1]." in output
    assert "* [E1] a.md > Licensing" in output
    assert "  [E2] b.md > Licensing" in output


def test_cli_without_a_question_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert "Usage" in capsys.readouterr().out