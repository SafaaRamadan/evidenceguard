from research_agent.verifier import check_citations

VALID = {"E1", "E2", "E3"}


def test_copied_header_is_not_an_answer() -> None:
    report = check_citations("[E1] Qwen3 > Licensing", VALID)

    assert not report.passed
    assert report.issues == ["header_only"]


def test_header_with_dots_in_the_title_is_caught() -> None:
    assert check_citations("[E1] Llama 3.3 > Overview", VALID).issues == ["header_only"]


def test_header_followed_by_uncited_sentences_is_an_uncited_sentence_problem() -> None:
    answer = "[E1] DeepSeek-R1 > Limitations\nThe model had repetitive outputs."

    report = check_citations(answer, VALID)

    assert report.issues == ["uncited_sentence"]


def test_a_normal_one_line_answer_is_not_header_only() -> None:
    assert check_citations("Qwen3 uses Apache 2.0 [E1].", VALID).passed


def test_a_sentence_containing_a_comparison_sign_is_not_header_only() -> None:
    report = check_citations("Model A > Model B on speed, according to the notes [E1].", VALID)

    assert report.passed