"""Score pipeline answers against benchmark questions and summarize them."""

from research_agent.pipeline import RagAnswer

def score_answer(question: dict, answer: RagAnswer) -> dict:
    """Turn one answer into a flat, JSON-friendly record."""
    gold = {(g["source"], g["section"]) for g in question["gold"]}
    evidence_pairs = {(hit.source, hit.section) for hit in answer.evidence.values()}
    cited_pairs = {
        (answer.evidence[evidence_id].source, answer.evidence[evidence_id].section)
        for evidence_id in answer.citations.cited_ids
        if evidence_id in answer.evidence
    }
    has_gold = bool(gold)
    return {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "answer": answer.text,
        "abstained": answer.abstained,
        "citation_issues": list(answer.citations.issues),
        "cited_ids": list(answer.citations.cited_ids),
        "has_gold": has_gold,
        "gold_in_evidence": bool(gold & evidence_pairs) if has_gold else None,
        "cited_gold": bool(gold & cited_pairs) if has_gold else None,
        "prompt_tokens": answer.prompt_tokens,
        "new_tokens": answer.new_tokens,
        "retrieval_ms": round(answer.retrieval_seconds * 1000, 1),
        "generation_s": round(answer.generation_seconds, 2),
    }


def summarize_answers(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["has_gold"]]
    unanswerable = [row for row in rows if not row["has_gold"]]
    answered = [row for row in answerable if not row["abstained"]]
    correct_abstentions = sum(1 for row in unanswerable if row["abstained"])
    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "answered": len(answered),
        "false_abstentions": len(answerable) - len(answered),
        "false_abstentions_gold_was_retrieved": sum(
            1 for row in answerable if row["abstained"] and row["gold_in_evidence"]
        ),
        "answered_with_clean_citations": sum(
            1 for row in answered if not row["citation_issues"]
        ),
        "answered_citing_gold": sum(1 for row in answered if row["cited_gold"]),
        "unanswerable": len(unanswerable),
        "correct_abstentions": correct_abstentions,
        "answered_when_unanswerable": len(unanswerable) - correct_abstentions,
        "structural_pass": sum(1 for row in rows if not row["citation_issues"]),
    }