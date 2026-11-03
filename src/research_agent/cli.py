"""Ask one question from the command line.

Usage:  python -m research_agent.cli "Under what license is Qwen3 released?"
"""

import logging
import sys

from research_agent.agent import build_agent


def main(argv: list[str] | None = None) -> int:
    question = " ".join(sys.argv[1:] if argv is None else argv).strip()
    if not question:
        print('Usage: python -m research_agent.cli "your question"')
        return 2

    logging.basicConfig(level=logging.WARNING)
    answer = build_agent().answer(question)

    print(f"\nQ: {answer.question}")
    print(f"Status: {answer.status} (attempts: {answer.attempts}, {answer.total_seconds:.1f} s)")
    print(f"\nA: {answer.text}\n")
    if answer.citations.issues:
        print("Issues:", ", ".join(answer.citations.issues))
    print("Evidence (* = cited):")
    cited = set(answer.citations.cited_ids)
    for evidence_id, hit in answer.evidence.items():
        mark = "*" if evidence_id in cited else " "
        print(f" {mark} [{evidence_id}] {hit.source} > {hit.section} ({hit.score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())