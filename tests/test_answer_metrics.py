import pytest
from research_agent.answer_metrics import score_answer, summarize_answers
from research_agent.index_store import SearchResult
from research_agent.pipeline import RagAnswer
from research_agent.prompting import ABSTAIN_MESSAGE
from research_agent.verifier import check_citations

GOLD = ("a.md", "Licensing")
OTHER = ("b.md", "Licensing")

ANSWERABLE = {
    "id": "q1",
    "category": "answerable",
    "question": "Which license?",
    "gold": [{"source": "a.md", "section": "Licensing"}],
}
ABSENT = {"id": "q2", "category": "absent", "question": "How much?", "gold": []}


def make_hit(pair: tuple[str, str]) -> SearchResult:
    source, section = pair
    return SearchResult(
        chunk_id=f"{source}:{section}",
        source=source,
        title="Title",
        section=section,
        text="text",
        source_url=None,
        date_accessed=None,
        score=0.5,
    )


def make_answer(text: str, evidence_pairs: list[tuple[str, str]]) -> RagAnswer:
    evidence = {
        f"E{rank}": make_hit(pair) for rank, pair in enumerate(evidence_pairs, start=1)
    }
    report = check_citations(text, set(evidence))
    return RagAnswer(
        question="q",
        text=text,
        abstained=report.abstained,
        evidence=evidence,
        citations=report,
        prompt_tokens=100,
        new_tokens=10,
        retrieval_seconds=0.02,
        generation_seconds=2.0,
    )


def test_answer_citing_the_gold_section() -> None:
    row = score_answer(ANSWERABLE, make_answer("Apache 2.0 [E1].", [GOLD, OTHER]))

    assert row["gold_in_evidence"] is True
    assert row["cited_gold"] is True
    assert row["abstained"] is False
    assert row["citation_issues"] == []


def test_answer_citing_a_different_chunk() -> None:
    row = score_answer(ANSWERABLE, make_answer("Apache 2.0 [E2].", [GOLD, OTHER]))

    assert row["gold_in_evidence"] is True
    assert row["cited_gold"] is False


def test_gold_section_not_retrieved() -> None:
    row = score_answer(ANSWERABLE, make_answer("Apache 2.0 [E1].", [OTHER]))

    assert row["gold_in_evidence"] is False
    assert row["cited_gold"] is False


def test_false_abstention_is_visible() -> None:
    row = score_answer(ANSWERABLE, make_answer(ABSTAIN_MESSAGE, [OTHER, GOLD]))

    assert row["abstained"] is True
    assert row["gold_in_evidence"] is True


def test_unanswerable_question_has_no_gold_fields() -> None:
    row = score_answer(ABSENT, make_answer(ABSTAIN_MESSAGE, [GOLD]))

    assert row["has_gold"] is False
    assert row["gold_in_evidence"] is None
    assert row["cited_gold"] is None


def test_invented_id_does_not_count_as_citing_gold() -> None:
    row = score_answer(ANSWERABLE, make_answer("Apache 2.0 [E9].", [GOLD]))

    assert row["cited_gold"] is False
    assert "invalid_citation" in row["citation_issues"]


def test_timings_and_token_counts_are_recorded() -> None:
    row = score_answer(ANSWERABLE, make_answer("Apache 2.0 [E1].", [GOLD]))

    assert row["retrieval_ms"] == pytest.approx(20.0)
    assert row["generation_s"] == pytest.approx(2.0)
    assert row["prompt_tokens"] == 100
    assert row["new_tokens"] == 10


def test_summary_counts() -> None:
    rows = [
        score_answer(ANSWERABLE, make_answer("Apache [E1].", [GOLD])),  # good
        score_answer(ANSWERABLE, make_answer(ABSTAIN_MESSAGE, [GOLD])),  # false abstention
        score_answer(ANSWERABLE, make_answer("[E1]", [GOLD])),  # citation only
        score_answer(ABSENT, make_answer(ABSTAIN_MESSAGE, [GOLD])),  # correct abstention
        score_answer(ABSENT, make_answer("Made up [E1].", [GOLD])),  # invented answer
    ]

    summary = summarize_answers(rows)

    assert summary["questions"] == 5
    assert summary["answerable"] == 3
    assert summary["answered"] == 2
    assert summary["false_abstentions"] == 1
    assert summary["false_abstentions_gold_was_retrieved"] == 1
    assert summary["answered_with_clean_citations"] == 1
    assert summary["answered_citing_gold"] == 2
    assert summary["unanswerable"] == 2
    assert summary["correct_abstentions"] == 1
    assert summary["answered_when_unanswerable"] == 1
    assert summary["structural_pass"] == 4  # the citation-only answer fails


def test_summary_of_no_rows_is_all_zeros() -> None:
    summary = summarize_answers([])

    assert summary["questions"] == 0
    assert summary["answered"] == 0