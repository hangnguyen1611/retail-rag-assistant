import asyncio
import csv
import json
import os
import re
import sys
import time

# Cho phép chạy trực tiếp "python src/eval/run_eval.py" từ project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from groq import AsyncGroq

from src.config import (
    EVAL_SET_PATH,
    EVAL_RESULTS_PATH,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    set_seed,
)
from src.rag.retriever import Retriever
from src.rag.generator import Generator

JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

REFUSAL_PATTERNS = [
    # Tiếng Việt — theo đúng câu chữ được yêu cầu trong prompt.py (SYSTEM_PROMPT_VI)
    "không có thông tin",
    "không tìm thấy thông tin",
    "không có thông tin này",
    "liên hệ nhân viên",
    "liên hệ nhân viên hỗ trợ",
    "không thể trả lời",
    "không nằm trong",
    # English — theo SYSTEM_PROMPT_EN
    "does not contain",
    "don't have this information",
    "do not have this information",
    "i don't have information",
    "contact support",
    "contact our support",
    "i'm not able to answer",
    "outside the scope",
]


def is_refusal(answer: str) -> bool:
    if not answer:
        return False
    text = answer.lower()
    return any(p in text for p in REFUSAL_PATTERNS)


def load_eval_set(path: str = EVAL_SET_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_relevant_ids(raw: str):
    if not raw:
        return set()
    return {x.strip() for x in raw.split(";") if x.strip()}


def _is_hit(hit: dict, doc_type: str, expected_ids: set) -> bool:
    metadata = hit.get("metadata", {})
    if doc_type == "product":
        return metadata.get("doc_type") == "product" and metadata.get("doc_id") in expected_ids
    if doc_type == "policy":
        return metadata.get("doc_type") == "policy" and metadata.get("source_file") in expected_ids
    return False


def compute_recall_at_k(retriever, eval_set, k_values=(1, 3)):
    """Recall@k trên các câu product/policy (bỏ qua out_of_scope).
    Trả về (summary_dict, per_row_hits) — per_row_hits dùng lại cho results.csv.
    """
    max_k = max(k_values)
    hit_counts = {k: 0 for k in k_values}
    total = 0
    per_row_hits = {}

    for row in eval_set:
        if row["category"] not in ("product", "policy"):
            continue
        total += 1
        expected_ids = _parse_relevant_ids(row["expected_relevant_ids"])
        hits = retriever.search(row["question"], top_k=max_k)
        per_row_hits[row["id"]] = hits

        for k in k_values:
            top_k_hits = hits[:k]
            if any(_is_hit(h, row["expected_doc_type"], expected_ids) for h in top_k_hits):
                hit_counts[k] += 1

    summary = {f"recall@{k}": (hit_counts[k] / total if total else 0.0) for k in k_values}
    summary["retrieval_eval_count"] = total
    return summary, per_row_hits


JUDGE_PROMPT_TEMPLATE = """You are grading a customer-support RAG assistant's answer for a fashion retail shop.

Question: {question}
Reference (key points the answer should cover, ground truth): {expected}
Assistant's answer: {answer}

Score the assistant's answer from 1 (worst) to 5 (best) on each dimension:
- relevance: does it actually address the question?
- correctness: does it match the reference key points (facts, numbers)?
- faithfulness: does it avoid inventing information not supported by context?

Respond with ONLY a JSON object, no other text, in this exact format:
{{"relevance": <1-5>, "correctness": <1-5>, "faithfulness": <1-5>}}
"""


async def llm_as_judge(client: AsyncGroq, query: str, answer: str, expected: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=query, expected=expected, answer=answer)
    try:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        scores = json.loads(raw)
        return {
            "relevance": float(scores.get("relevance", 0)),
            "correctness": float(scores.get("correctness", 0)),
            "faithfulness": float(scores.get("faithfulness", 0)),
        }
    except Exception as e:
        print(f"  [judge parse error] {e}")
        return {"relevance": None, "correctness": None, "faithfulness": None}


async def run_generation_and_judge(eval_set, retriever, generator, judge_client):
    rows_out = []
    for row in eval_set:
        qid, question, category = row["id"], row["question"], row["category"]
        language = row.get("language", "auto") or "auto"

        hits = retriever.search(question, top_k=TOP_K)
        context = "\n\n---\n\n".join(h["content"] for h in hits)

        try:
            gen = await generator.generate(question, context, language=language)
            answer = gen["answer"]
            latency_ms = gen["latency_ms"]
        except Exception as e:
            print(f"  [generation error] {qid}: {e}")
            answer, latency_ms = "", None

        result_row = {
            "id": qid,
            "category": category,
            "question": question,
            "answer": answer,
            "latency_ms": latency_ms,
            "retrieved_ids": ";".join(h["metadata"].get("doc_id", h["id"]) for h in hits),
        }

        if category == "out_of_scope":
            result_row["refusal_correct"] = is_refusal(answer)
            result_row["relevance"] = result_row["correctness"] = result_row["faithfulness"] = None
        else:
            scores = await llm_as_judge(judge_client, question, answer, row["expected_answer_keypoints"])
            result_row.update(scores)
            result_row["refusal_correct"] = None

        rows_out.append(result_row)
        print(f"  done {qid} ({category})")

    return rows_out


def save_results(rows, path: str = EVAL_RESULTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "id", "category", "question", "answer", "latency_ms", "retrieved_ids",
        "relevance", "correctness", "faithfulness", "refusal_correct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_summary(recall_summary, gen_rows):
    scored = [r for r in gen_rows if r["category"] != "out_of_scope"]
    oos = [r for r in gen_rows if r["category"] == "out_of_scope"]
    latencies = [r["latency_ms"] for r in gen_rows if r["latency_ms"] is not None]

    def avg(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    print(f"Retrieval Recall@1:      {recall_summary['recall@1']:.2%}  (n={recall_summary['retrieval_eval_count']})")
    print(f"Retrieval Recall@3:      {recall_summary['recall@3']:.2%}  (n={recall_summary['retrieval_eval_count']})")
    print(f"Answer relevance (1-5):  {avg('relevance', scored)}")
    print(f"Answer correctness(1-5): {avg('correctness', scored)}")
    print(f"Answer faithfulness(1-5):{avg('faithfulness', scored)}")
    if latencies:
        print(f"Avg latency (ms):        {sum(latencies)/len(latencies):.1f}")
    if oos:
        refusal_acc = sum(1 for r in oos if r["refusal_correct"]) / len(oos)
        print(f"Refusal accuracy:        {refusal_acc:.2%}  (n={len(oos)})")
    print("=" * 50)
    print(f"Detailed results saved to {EVAL_RESULTS_PATH}")


async def main_async():
    set_seed()
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval questions")

    retriever = Retriever(persist_dir=CHROMA_PERSIST_DIR, embedding_model=EMBEDDING_MODEL, top_k=TOP_K)
    generator = Generator()
    judge_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    print("Computing retrieval recall@1 / recall@3 ...")
    recall_summary, _ = compute_recall_at_k(retriever, eval_set, k_values=(1, 3))

    print("Running generation + LLM-as-judge for each question ...")
    gen_rows = await run_generation_and_judge(eval_set, retriever, generator, judge_client)

    save_results(gen_rows)
    print_summary(recall_summary, gen_rows)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
