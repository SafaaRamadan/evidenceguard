import hashlib
import json
import sys
from pathlib import Path
import psutil
from research_agent.answer_metrics import score_answer, summarize_answers
from research_agent.chunking import MAX_WORDS, OVERLAP_WORDS
from research_agent.embeddings import DEFAULT_MODEL_NAME, Embedder
from research_agent.generator import DEFAULT_LLM_NAME, Generator
from research_agent.index_store import IndexStore
from research_agent.pipeline import RagPipeline
from research_agent.prompting import SYSTEM_PROMPT
from research_agent.retriever import Retriever, build_index

QUESTIONS_PATH = Path("evaluation/questions.json")
RESULTS_DIR = Path("evaluation/results")
DATA_DIR = Path("data/raw")
EVAL_DB_PATH = Path("chroma_db") / "e2e_eval"
TOP_K = 3
MAX_NEW_TOKENS = 150
WARMUP_QUESTION = "Which license does Qwen3 use?"


def peak_memory_mb() -> float:
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1024**2


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def flag(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def print_table(rows: list[dict]) -> None:
    print(
        f"{'id':<5}{'category':<13}{'outcome':<11}{'gold in ev.':<13}"
        f"{'cites gold':<12}{'prompt':>7}{'gen s':>7}  issues"
    )
    for row in rows:
        outcome = "ABSTAINED" if row["abstained"] else "ANSWERED"
        issues = ",".join(row["citation_issues"]) or "-"
        print(
            f"{row['id']:<5}{row['category']:<13}{outcome:<11}"
            f"{flag(row['gold_in_evidence']):<13}{flag(row['cited_gold']):<12}"
            f"{row['prompt_tokens']:>7}{row['generation_s']:>7.1f}  {issues}"
        )


def print_answers(rows: list[dict]) -> None:
    print("\nANSWERS (read these and judge them yourself)")
    for row in rows:
        print(f"\n{row['id']} [{row['category']}] {row['question']}")
        print(f"   -> {row['answer']}")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "e2e_v1"
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    embedder = Embedder()
    store = IndexStore(EVAL_DB_PATH)
    chunk_count = build_index(DATA_DIR, embedder, store)
    generator = Generator()
    pipeline = RagPipeline(
        Retriever(embedder, store),
        generator,
        top_k=TOP_K,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    pipeline.answer(WARMUP_QUESTION) 

    rows = []
    for question in questions:
        rows.append(score_answer(question, pipeline.answer(question["question"])))
        print(f"finished {question['id']}", flush=True)

    summary = summarize_answers(rows)
    retrieval_ms = mean([row["retrieval_ms"] for row in rows])
    generation_s = [row["generation_s"] for row in rows]
    performance = {
        "retrieval_ms_mean": round(retrieval_ms, 1),
        "generation_s_mean": round(mean(generation_s), 2),
        "generation_s_max": round(max(generation_s), 2),
        "total_s_mean": round(retrieval_ms / 1000 + mean(generation_s), 2),
        "prompt_tokens_mean": round(mean([row["prompt_tokens"] for row in rows])),
        "peak_memory_mb": round(peak_memory_mb()),
    }

    print()
    print_table(rows)
    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nPERFORMANCE")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    print_answers(rows)

    output = {
        "name": name,
        "config": {
            "embedding_model": DEFAULT_MODEL_NAME,
            "llm": DEFAULT_LLM_NAME,
            "llm_dtype": generator.dtype_name,
            "max_words": MAX_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "chunk_count": chunk_count,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8],
            "system_prompt": SYSTEM_PROMPT,
        },
        "summary": summary,
        "performance": performance,
        "per_question": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{name}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()