from research_agent.prompting import ABSTAIN_MESSAGE
from research_agent.verifier import check_citations, extract_citations

VALID = {"E1", "E2"}


def test_well_formed_answer_passes() -> None:
    report = check_citations("Qwen3 uses Apache 2.0 [E1].", VALID)

    assert report.passed
    assert report.cited_ids == ["E1"]
    assert report.issues == []


def test_multiple_and_comma_separated_citations() -> None:
    report = check_citations("A is fast [E1]. B is small [E1, E2].", VALID)

    assert report.passed
    assert report.cited_ids == ["E1", "E2"]


def test_invalid_id_is_flagged() -> None:
    report = check_citations("X is true [E7].", VALID)

    assert not report.passed
    assert report.invalid_ids == ["E7"]
    assert report.issues == ["invalid_citation"]


def test_answer_without_citations_is_flagged() -> None:
    report = check_citations("Qwen uses Apache.", VALID)

    assert report.issues == ["no_citations"]
    assert report.uncited_sentences == ["Qwen uses Apache."]


def test_citation_only_answer_has_no_content() -> None:
    report = check_citations("[E1]", VALID)

    assert report.cited_ids == ["E1"]
    assert report.issues == ["no_content"]


def test_uncited_sentence_next_to_cited_one_is_flagged() -> None:
    report = check_citations("A is fast [E1]. B is small.", VALID)

    assert report.issues == ["uncited_sentence"]
    assert report.uncited_sentences == ["B is small."]


def test_exact_abstention_passes_without_citations() -> None:
    report = check_citations(ABSTAIN_MESSAGE, VALID)

    assert report.abstained
    assert report.passed
    assert report.cited_ids == []


def test_abstention_with_surrounding_whitespace_is_recognized() -> None:
    assert check_citations(f"  {ABSTAIN_MESSAGE}\n", VALID).abstained


def test_empty_answer_is_flagged() -> None:
    report = check_citations("   ", VALID)

    assert not report.passed
    assert report.issues == ["empty_answer"]


def test_extract_citations_keeps_order_and_drops_duplicates() -> None:
    assert extract_citations("a [E2] b [E1] c [E2]") == ["E2", "E1"]


def test_lookalike_brackets_are_not_citations() -> None:
    assert extract_citations("See [note], [E] and [1].") == []


def test_passing_does_not_prove_the_claim_is_true() -> None:
    # The evidence might say Apache 2.0, but the structural check cannot know.
    report = check_citations("Qwen3 is licensed under GPL [E1].", VALID)

    assert report.passed