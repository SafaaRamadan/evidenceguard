import re
from dataclasses import dataclass

from research_agent.prompting import ABSTAIN_MESSAGE

CITATION_PATTERN = re.compile(r"\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]")
ID_PATTERN = re.compile(r"E\d+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_HEADER_WORDS = 12
HAS_WORD = re.compile(r"\w")


@dataclass(frozen=True)
class CitationReport:
    """What the structural checks found in one answer."""

    abstained: bool
    cited_ids: list[str]  # unique, in order of first appearance
    invalid_ids: list[str]  # cited IDs that were never offered as evidence
    uncited_sentences: list[str]
    issues: list[str]  # codes such as "no_citations"; empty means no problems

    @property
    def passed(self) -> bool:
        return not self.issues


def extract_citations(answer: str) -> list[str]:
    """Return the unique evidence IDs cited in `answer`, in order."""
    ids: list[str] = []
    for match in CITATION_PATTERN.finditer(answer):
        for evidence_id in ID_PATTERN.findall(match.group(1)):
            if evidence_id not in ids:
                ids.append(evidence_id)
    return ids


def strip_citations(answer: str) -> str:
    """Remove every citation tag, leaving the rest of the text."""
    return CITATION_PATTERN.sub("", answer)


def _is_header_only(text: str) -> bool:
    """True for a lone "Title > Section" line copied from the evidence, not an answer."""
    return (
        "\n" not in text
        and " > " in text
        and len(text.split()) <= MAX_HEADER_WORDS
        and text[-1] not in ".!?"
    )

def check_citations(answer: str, valid_ids: set[str]) -> CitationReport:
    """Check an answer's citations against the evidence IDs that were offered."""
    answer = answer.strip()
    if not answer:
        return CitationReport(False, [], [], [], ["empty_answer"])
    if answer == ABSTAIN_MESSAGE:
        return CitationReport(True, [], [], [], [])

    cited = extract_citations(answer)
    invalid = [evidence_id for evidence_id in cited if evidence_id not in valid_ids]

    uncited = []
    for sentence in SENTENCE_SPLIT.split(answer):
        sentence = sentence.strip()
        has_text = HAS_WORD.search(strip_citations(sentence))
        if has_text and not CITATION_PATTERN.search(sentence):
            uncited.append(sentence)

    remainder = strip_citations(answer).strip()
    issues = []
    if not HAS_WORD.search(remainder):
        issues.append("no_content")
    elif _is_header_only(remainder):
        issues.append("header_only")
    if not cited:
        issues.append("no_citations")
    if invalid:
        issues.append("invalid_citation")
    if cited and uncited:
        issues.append("uncited_sentence")

    return CitationReport(False, cited, invalid, uncited, issues)