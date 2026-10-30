import json
import sys
import time
from pathlib import Path

from research_agent.chunking import MAX_WORDS, OVERLAP_WORDS
from research_agent.embeddings import DEFAULT_MODEL_NAME, Embedder
from research_agent.index_store import IndexStore
from research_agent.retrieval_metrics import hit_at_k, recall_at_k, reciprocal_rank
from research_agent.retriever import Retriever, build_index

QUESTIONS_PATH = Path("evaluation/questions.json")
RESULTS_DIR = Path("evaluation/results")
DATA_DIR = Path("data/raw")
EVAL_DB_PATH = Path("chroma_db") / "eval"
TOP_K = 3


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("No scored questions to average.")
    return sum(values) / len(values)


def evaluate(questions: list[dict], retriever: Retriever) -> list[dict]:
    """Retrieve for every question and compute its metrics."""
    rows = []
    for question in questions:
        start = time.perf_counter()
        hits = retriever.retrieve(question["question"], top_k=TOP_K)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved = [(hit.source, hit.section) for hit in hits]
        gold = [(g["source"], g["section"]) for g in question["gold"]]

        row = {
            "id": question["id"],
            "category": question["category"],
            "latency_ms": round(latency_ms, 1),
            "top_score": round(hits[0].score, 3),
            "retrieved": [list(pair) for pair in retrieved],
        }
        if gold:  # absent and ambiguous questions have no gold, so no metrics
            row["hit_at_1"] = hit_at_k(retrieved, gold, 1)
            row["hit_at_3"] = hit_at_k(retrieved, gold, 3)
            row["recall_at_3"] = recall_at_k(retrieved, gold, 3)
            row["reciprocal_rank"] = reciprocal_rank(retrieved, gold)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Average the metrics over the questions that have a gold answer."""
    scored = [row for row in rows if "hit_at_1" in row]
    return {
        "n": len(scored),
        "hits_at_1": int(sum(row["hit_at_1"] for row in scored)),
        "hits_at_3": int(sum(row["hit_at_3"] for row in scored)),
        "hit_at_1": mean([row["hit_at_1"] for row in scored]),
        "hit_at_3": mean([row["hit_at_3"] for row in scored]),
        "recall_at_3": mean([row["recall_at_3"] for row in scored]),
        "mrr": mean([row["reciprocal_rank"] for row in scored]),
    }


def print_table(rows: list[dict]) -> None:
    print(f"{'id':<5}{'category':<13}{'H@1':>4}{'H@3':>4}{'R@3':>6}{'RR':>6}{'ms':>7}  top result")
    for row in rows:
        top_source, top_section = row["retrieved"][0]
        if "hit_at_1" in row:
            metrics = (
                f"{row['hit_at_1']:>4.0f}{row['hit_at_3']:>4.0f}"
                f"{row['recall_at_3']:>6.2f}{row['reciprocal_rank']:>6.2f}"
            )
        else:
            metrics = f"{'-':>4}{'-':>4}{'-':>6}{'-':>6}"
        print(
            f"{row['id']:<5}{row['category']:<13}{metrics}{row['latency_ms']:>7.1f}"
            f"  {top_source} > {top_section} ({row['top_score']:.3f})"
        )


def print_summary(label: str, summary: dict) -> None:
    n = summary["n"]
    print(
        f"{label:<12} n={n}  "
        f"Hit@1={summary['hit_at_1']:.2f} ({summary['hits_at_1']}/{n})  "
        f"Hit@3={summary['hit_at_3']:.2f} ({summary['hits_at_3']}/{n})  "
        f"Recall@3={summary['recall_at_3']:.2f}  MRR={summary['mrr']:.2f}"
    )


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    embedder = Embedder()
    store = IndexStore(EVAL_DB_PATH)
    chunk_count = build_index(DATA_DIR, embedder, store)
    retriever = Retriever(embedder, store)
    embedder.embed_query("warm up")  # first call is slower; keep it out of timings

    rows = evaluate(questions, retriever)
    overall = summarize(rows)
    categories = sorted({row["category"] for row in rows if "hit_at_1" in row})
    by_category = {
        category: summarize([row for row in rows if row["category"] == category])
        for category in categories
    }

    print_table(rows)
    print(f"\nChunks indexed: {chunk_count}, k = {TOP_K}")
    print_summary("OVERALL", overall)
    for category, summary in by_category.items():
        print_summary(category, summary)
    latencies = [row["latency_ms"] for row in rows]
    print(f"Latency: mean {mean(latencies):.1f} ms, max {max(latencies):.1f} ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "name": name,
        "config": {
            "embedding_model": DEFAULT_MODEL_NAME,
            "max_words": MAX_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "top_k": TOP_K,
            "chunk_count": chunk_count,
        },
        "summary": {"overall": overall, "by_category": by_category},
        "per_question": rows,
    }
    output_path = RESULTS_DIR / f"{name}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()