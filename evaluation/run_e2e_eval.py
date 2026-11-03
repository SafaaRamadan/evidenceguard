"""Run the whole system on benchmark questions and save the results.

Examples (from the project root):
  python evaluation/run_e2e_eval.py e2e_v1
  python evaluation/run_e2e_eval.py e2e_agent --mode agent
  python evaluation/run_e2e_eval.py e2e_agent_retry --mode agent --retry-on-abstain
  python evaluation/run_e2e_eval.py adv_agent --mode agent \
      --extra-data data/adversarial --questions evaluation/questions_adversarial.json
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import psutil

from research_agent.agent import EvidenceAgent
from research_agent.answer_metrics import score_answer, summarize_answers
from research_agent.chunking import MAX_WORDS, OVERLAP_WORDS
from research_agent.embeddings import DEFAULT_MODEL_NAME, Embedder
from research_agent.generator import DEFAULT_DTYPE, DEFAULT_LLM_NAME, Generator
from research_agent.index_store import IndexStore
from research_agent.pipeline import RagPipeline
from research_agent.prompting import SYSTEM_PROMPT
from research_agent.retriever import Retriever, build_index

RESULTS_DIR = Path("evaluation/results")
DATA_DIR = Path("data/raw")
DEFAULT_QUESTIONS = Path("evaluation/questions.json")
EVAL_DB_PATH = Path("chroma_db") / "e2e_eval"
TOP_K = 3
MAX_NEW_TOKENS = 150
WARMUP_QUESTION = "Which license does Qwen3 use?"


def memory_mb() -> dict:
    info = psutil.Process().memory_info()
    return {
        "rss_mb": round(info.rss / 1024**2),
        "peak_working_set_mb": round(getattr(info, "peak_wset", info.rss) / 1024**2),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def flag(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def status_of(answer) -> str:
    """Agent answers carry a status; pipeline answers get the same labels."""
    if hasattr(answer, "status"):
        return answer.status
    if answer.abstained:
        return "abstained"
    return "answered" if answer.citations.passed else "unverified"


def collect_corpus(directories: list[Path], target: Path) -> None:
    """Copy the documents of several folders into one, refusing duplicate names."""
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Data directory not found: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file():
                destination = target / path.name
                if destination.exists():
                    raise ValueError(f"Duplicate document name across folders: {path.name}")
                shutil.copy(path, destination)


def evaluate(questions: list[dict], answerer) -> list[dict]:
    rows = []
    for question in questions:
        answer = answerer.answer(question["question"])
        row = score_answer(question, answer)
        row["status"] = status_of(answer)
        row["attempts"] = getattr(answer, "attempts", 1)
        row["trace"] = list(getattr(answer, "trace", ()))
        row["generation_s"] = round(answer.generation_seconds, 2)
        if "canary" in question:
            row["canary_leaked"] = question["canary"] in answer.text
        rows.append(row)
        print(f"finished {question['id']}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = summarize_answers(rows)
    summary["unverified_answers"] = sum(1 for row in rows if row["status"] == "unverified")
    summary["mean_attempts"] = round(mean([row["attempts"] for row in rows]), 2)
    canary_rows = [row for row in rows if "canary_leaked" in row]
    if canary_rows:
        summary["injection_questions"] = len(canary_rows)
        summary["injection_leaks"] = sum(1 for row in canary_rows if row["canary_leaked"])
    return summary


def print_table(rows: list[dict]) -> None:
    print(
        f"{'id':<5}{'category':<13}{'status':<11}{'tries':>5}{'gold in ev.':>12}"
        f"{'cites gold':>11}{'gen s':>7}  issues"
    )
    for row in rows:
        issues = ",".join(row["citation_issues"]) or "-"
        print(
            f"{row['id']:<5}{row['category']:<13}{row['status']:<11}{row['attempts']:>5}"
            f"{flag(row['gold_in_evidence']):>12}{flag(row['cited_gold']):>11}"
            f"{row['generation_s']:>7.1f}  {issues}"
        )


def print_answers(rows: list[dict]) -> None:
    print("\nANSWERS (read these and judge them yourself)")
    for row in rows:
        print(f"\n{row['id']} [{row['category']}] {row['question']}")
        print(f"   -> {row['answer']}")
        if row["trace"]:
            print(f"   trace: {' | '.join(row['trace'])}")
        if row.get("canary_leaked"):
            print("   !! the injected canary text appears in this answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("name", help="name for this run; results go to evaluation/results/<name>.json")
    parser.add_argument("--mode", choices=["pipeline", "agent"], default="pipeline")
    parser.add_argument("--retry-on-abstain", action="store_true", help="agent mode: widen retrieval after an abstention")
    parser.add_argument("--extra-data", type=Path, action="append", default=[], help="extra document folder to add to data/raw")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=TOP_K, help="evidence passages shown to the model")
    parser.add_argument("--llm", default=DEFAULT_LLM_NAME)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["float32", "bfloat16"])
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing results file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = RESULTS_DIR / f"{args.name}.json"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} already exists. Pick a new name or pass --overwrite.")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))

    embedder = Embedder()
    store = IndexStore(EVAL_DB_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        collect_corpus([DATA_DIR, *args.extra_data], corpus)
        chunk_count = build_index(corpus, embedder, store)

    generator = Generator(model_name=args.llm, dtype_name=args.dtype)
    retriever = Retriever(embedder, store)
    if args.mode == "agent":
        answerer = EvidenceAgent(
            retriever, generator, top_k=args.top_k, widen_k=max(5, args.top_k + 2),
            max_new_tokens=MAX_NEW_TOKENS, retry_on_abstain=args.retry_on_abstain,
        )
    else:
        answerer = RagPipeline(
            retriever, generator, top_k=args.top_k, max_new_tokens=MAX_NEW_TOKENS
        )

    answerer.answer(WARMUP_QUESTION)  # first call is slower; keep it out of the results

    rows = evaluate(questions, answerer)
    summary = summarize(rows)
    generation = [row["generation_s"] for row in rows]
    retrieval_ms = mean([row["retrieval_ms"] for row in rows])
    performance = {
        "retrieval_ms_mean": round(retrieval_ms, 1),
        "generation_s_mean": round(mean(generation), 2),
        "generation_s_max": round(max(generation), 2),
        "total_s_mean": round(retrieval_ms / 1000 + mean(generation), 2),
        "prompt_tokens_mean": round(mean([row["prompt_tokens"] for row in rows])),
        **memory_mb(),
    }

    print()
    print_table(rows)
    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nPERFORMANCE (memory numbers are indicative, not exact)")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    print_answers(rows)

    output = {
        "name": args.name,
        "config": {
            "mode": args.mode,
            "retry_on_abstain": args.retry_on_abstain,
            "embedding_model": DEFAULT_MODEL_NAME,
            "llm": args.llm,
            "llm_dtype": args.dtype,
            "max_words": MAX_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "top_k": args.top_k,
            "max_new_tokens": MAX_NEW_TOKENS,
            "chunk_count": chunk_count,
            "questions_file": str(args.questions),
            "extra_data": [str(path) for path in args.extra_data],
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8],
            "system_prompt": SYSTEM_PROMPT,
        },
        "summary": summary,
        "performance": performance,
        "per_question": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()