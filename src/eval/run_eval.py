"""
run_eval.py

Chạy eval trên data/eval/eval_set.csv (build bởi src/eval/build_eval_set.py).

Thay đổi so với bản cũ:
1. JUDGE_MODEL không còn im lặng fallback về GROQ_MODEL -> model không tự chấm
   chính nó nữa (self-preference bias làm điểm bị đội lên).
2. Judge prompt strictness-aware: câu `strict` chấm theo đúng 1 sản phẩm, câu
   `loose` chấm "có nêu được ít nhất một sản phẩm trong tập khớp".
3. Category mới `product_not_found`: đo hallucination trực tiếp (bịa ra sản
   phẩm không tồn tại), chấm bằng judge chứ không bằng regex.
4. Recall báo cáo TÁCH theo nhóm (strict / loose / policy) — gộp lại thì con số
   vô nghĩa vì strict và loose có độ khó khác nhau hẳn.
5. Kèm khoảng tin cậy 95% (Wilson) cho mọi tỷ lệ, vì n mỗi nhóm chỉ 15-30.

Chạy:
    python -m src.eval.run_eval
    EVAL_DRY_RUN=1 python -m src.eval.run_eval    # smoke test, không gọi Groq
"""

import asyncio
import csv
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (
    EVAL_SET_PATH,
    EVAL_RESULTS_PATH,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    TOP_K,
    set_seed,
)

DRY_RUN = os.getenv("EVAL_DRY_RUN") == "1"
RECALL_ONLY = os.getenv("EVAL_RECALL_ONLY") == "1"

# Dry run và recall-only KHÔNG được ghi đè kết quả thật — nếu dùng chung file thì
# một lần smoke test là mất baseline mà không có cảnh báo nào.
if DRY_RUN:
    RESULTS_PATH = EVAL_RESULTS_PATH.replace(".csv", "_dryrun.csv")
elif RECALL_ONLY:
    RESULTS_PATH = EVAL_RESULTS_PATH.replace(".csv", "_recall.csv")
else:
    RESULTS_PATH = EVAL_RESULTS_PATH

# Số câu xử lý song song. Gần như toàn bộ thời gian là chờ network, nên chạy
# tuần tự là lãng phí. Gặp nhiều lỗi 503 thì giảm xuống.
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "1"))

# --- Judge model: phải khai báo tường minh và phải KHÁC model sinh câu trả lời.
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "")
if not (DRY_RUN or RECALL_ONLY):   # hai chế độ này không gọi judge
    if not JUDGE_MODEL:
        sys.exit(
            "Chưa set GROQ_JUDGE_MODEL.\n"
            f"  Model sinh câu trả lời hiện tại: {GROQ_MODEL}\n"
            "  Judge phải là model khác, ví dụ:\n"
            "    export GROQ_JUDGE_MODEL=qwen/qwen3.6-27b\n"
            "  Để cùng model tự chấm chính nó sẽ làm điểm bị đội lên (self-preference bias)."
        )
    if JUDGE_MODEL == GROQ_MODEL:
        sys.exit(
            f"GROQ_JUDGE_MODEL == GROQ_MODEL ({JUDGE_MODEL}). "
            "Model không được tự chấm chính nó — chọn judge khác."
        )

# Regex chỉ còn dùng làm fallback khi judge parse lỗi.
REFUSAL_PATTERNS = [
    "không có thông tin", "không tìm thấy", "không có sản phẩm", "không có mẫu",
    "hiện không có", "chưa có", "liên hệ nhân viên", "không thể trả lời", "không nằm trong",
    "does not contain", "don't have this information", "do not have this information",
    "i don't have information", "we don't carry", "not available", "no matching",
    "contact support", "contact our support", "i'm not able to answer", "outside the scope",
]


def is_refusal(answer: str) -> bool:
    if not answer:
        return False
    text = answer.lower()
    return any(p in text for p in REFUSAL_PATTERNS)


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Khoảng tin cậy 95% cho tỷ lệ. Với n = 15-30 thì khoảng này rất rộng —
    in ra để không ai đọc '90%' như một con số chắc chắn."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


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


def _group_of(row: dict) -> str:
    """Nhóm để báo cáo riêng — strict và loose khó khác nhau hẳn nên không gộp."""
    if row["category"] == "product":
        return f"product/{row.get('strictness') or 'loose'}"
    return row["category"]


def retrieve_all(retriever, eval_set, top_k):
    """Retrieve MỘT lần cho toàn bộ eval set, dùng chung cho cả recall lẫn
    generation. Bản trước search 2 lần (80 + 105 = 185 lần encode)."""
    questions = [r["question"] for r in eval_set]
    if hasattr(retriever, "search_many"):
        all_hits = retriever.search_many(questions, top_k=top_k)
    else:
        all_hits = [retriever.search(q, top_k=top_k) for q in questions]
    return {r["id"]: h for r, h in zip(eval_set, all_hits)}


def compute_recall_at_k(hits_by_id, eval_set, k_values=(1, 3)):
    """Recall@k theo từng nhóm, tính trên hits đã retrieve sẵn."""
    groups = {}
    for row in eval_set:
        if row["category"] not in ("product", "policy"):
            continue
        group = _group_of(row)
        stats = groups.setdefault(group, {"total": 0, **{k: 0 for k in k_values}})
        stats["total"] += 1

        expected_ids = _parse_relevant_ids(row["expected_relevant_ids"])
        hits = hits_by_id[row["id"]]

        for k in k_values:
            if any(_is_hit(h, row["expected_doc_type"], expected_ids) for h in hits[:k]):
                stats[k] += 1
    return groups


# ------------------------------------------------------------------ judges
JUDGE_STRICT = """You are grading a customer-support RAG assistant for a fashion retail shop.

Question: {question}
Reference (the ONE product this question refers to, with its true facts): {expected}
Assistant's answer: {answer}

Score 1 (worst) to 5 (best):
- relevance: does it address the question?
- correctness: do the facts (price, stock, size, product identity) match the reference exactly?
- faithfulness: does it avoid stating anything not supported by the reference?

Respond with ONLY a JSON object:
{{"relevance": <1-5>, "correctness": <1-5>, "faithfulness": <1-5>}}
"""

JUDGE_LOOSE = """You are grading a customer-support RAG assistant for a fashion retail shop.

The question is deliberately broad — MANY catalog products match it, so there is no
single right answer. The reference below describes the whole matching set.

Question: {question}
Reference (the matching set): {expected}
Assistant's answer: {answer}

Score 1 (worst) to 5 (best):
- relevance: does it address the question?
- correctness: does it name at least one product consistent with the matching set, with
  price/stock figures that fall within the ranges given? Do NOT penalise it for naming a
  different valid product than the examples listed.
- faithfulness: does it avoid inventing products or figures outside the described set?

Respond with ONLY a JSON object:
{{"relevance": <1-5>, "correctness": <1-5>, "faithfulness": <1-5>}}
"""

JUDGE_REFUSAL = """You are grading whether a fashion-retail support assistant correctly
declined a question it had no data for.

Question: {question}
Why it should decline: {expected}
Assistant's answer: {answer}

Answer two things:
- refused: 1 if the assistant clearly says it cannot help / has no such product / no
  information, else 0. Offering a genuine alternative while still saying the requested
  item is unavailable still counts as 1.
- invented: 1 if the assistant asserts as fact a product, price, or stock figure that
  the question implies does not exist, else 0.

Respond with ONLY a JSON object:
{{"refused": <0 or 1>, "invented": <0 or 1>}}
"""


def _extract_json(raw):
    if not raw or not raw.strip():
        raise ValueError("judge trả về content RỖNG (model reasoning? thử reasoning_effort='none')")
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^<think>.*", "", text, flags=re.S | re.I)
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"không tìm thấy JSON trong: {text[:200]!r}")
    return json.loads(m.group(0))


MAX_RETRIES = 6


class DailyQuotaExhausted(RuntimeError):
    """Hết quota NGÀY (TPD/RPD) — retry vô nghĩa, phải dừng cả run."""


def _parse_retry_after(msg: str):
    """Groq nói thẳng cần chờ bao lâu. '1m23.376s' -> m trước chữ số là phút;
    '690ms' -> m trước 's' là milli."""
    m = re.search(r"try again in\s+(?:(\d+)m(?=[\d.]))?\s*([\d.]+)\s*(ms|s)\b", msg)
    if not m:
        return None
    val = float(m.group(2))
    if m.group(3) == "ms":
        val /= 1000.0
    return int(m.group(1) or 0) * 60 + val


async def _with_retry(coro_factory, label: str):
    """Retry cho giới hạn TẠM THỜI (TPM/RPM/503), tôn trọng đúng thời gian Groq
    yêu cầu. Gặp giới hạn NGÀY thì raise ngay — chờ cũng không hết."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_factory()
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            last = e

            if "rate_limit" in low and ("per day" in low or "tpd" in low or "rpd" in low):
                raise DailyQuotaExhausted(msg) from e

            transient = ("rate_limit" in low or "429" in msg
                         or "503" in msg or "over capacity" in low)
            if not transient:
                raise

            wait = _parse_retry_after(msg)
            if wait is None:
                wait = 2 ** attempt
            wait = min(wait + 0.5, 90)       # +0.5s đệm, chặn trần 90s
            print(f"      [{label}] rate limit, chờ {wait:.1f}s (lần {attempt + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait)
    raise last


async def _call_judge(client, prompt):
    kwargs = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    # qwen3 mặc định BẬT reasoning -> token suy luận vẫn bị tính vào quota NGÀY
    # dù `content` trả về rỗng. Đây là thứ đã đốt hết 200k TPD lần trước.
    if "qwen3" in JUDGE_MODEL.lower():
        kwargs["reasoning_effort"] = "none"

    async def call():
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    return await _with_retry(call, "judge")


async def judge_quality(client, row, answer):
    template = JUDGE_STRICT if row.get("strictness") == "strict" else JUDGE_LOOSE
    prompt = template.format(
        question=row["question"],
        expected=row["expected_answer_keypoints"],
        answer=answer,
    )
    raw = ""
    try:
        raw = await _call_judge(client, prompt)
        s = _extract_json(raw)
        return {k: float(s.get(k, 0)) for k in ("relevance", "correctness", "faithfulness")}
    except DailyQuotaExhausted:
        raise
    except Exception as e:
        print(f"  [judge error] {row['id']}: {e}")
        print(f"      raw: {raw[:200]!r}")
        return {"relevance": None, "correctness": None, "faithfulness": None}


async def judge_refusal(client, row, answer):
    """Không dùng regex ở đây: trợ lý có thể từ chối đúng bằng câu 'shop chưa có
    sandal màu xanh lá' — không khớp pattern nào nhưng hoàn toàn đúng."""
    prompt = JUDGE_REFUSAL.format(
        question=row["question"],
        expected=row["expected_answer_keypoints"] or "The question is outside the shop's domain.",
        answer=answer,
    )
    raw = ""
    try:
        raw = await _call_judge(client, prompt)
        s = _extract_json(raw)
        return {"refused": int(s.get("refused", 0)), "invented": int(s.get("invented", 0))}
    except DailyQuotaExhausted:
        raise
    except Exception as e:
        print(f"  [judge error] {row['id']}: {e} -> fallback regex")
        print(f"      raw: {raw[:200]!r}")
        return {"refused": int(is_refusal(answer)), "invented": None}


# ------------------------------------------------------------------ run
async def _process_row(row, hits, generator, judge_client, sem, counters):
    async with sem:
        qid, category = row["id"], row["category"]
        context = "\n\n---\n\n".join(h["content"] for h in hits)

        try:
            gen = await _with_retry(
                lambda: generator.generate(row["question"], context,
                                           language=row.get("language") or "auto"),
                "gen",
            )
            answer, latency_ms = gen["answer"], gen["latency_ms"]
        except DailyQuotaExhausted:
            raise
        except Exception as e:
            print(f"  [generation error] {qid}: {e}")
            answer, latency_ms = "", None

        out = {
            "id": qid,
            "category": category,
            "group": _group_of(row),
            "strictness": row.get("strictness", ""),
            "question": row["question"],
            "answer": answer,
            "latency_ms": latency_ms,
            "n_expected": len(_parse_relevant_ids(row["expected_relevant_ids"])),
            "retrieved_ids": ";".join(h["metadata"].get("doc_id", h["id"]) for h in hits),
            "relevance": None, "correctness": None, "faithfulness": None,
            "refused": None, "invented": None,
        }

        if category in ("out_of_scope", "product_not_found"):
            out.update(await judge_refusal(judge_client, row, answer))
        else:
            out.update(await judge_quality(judge_client, row, answer))

        if out["relevance"] is None and out["invented"] is None:
            counters["fail"] += 1
        else:
            counters["fail"] = 0
        counters["done"] += 1
        print(f"  done {counters['done']}/{counters['total']} {qid} ({out['group']})")
        return out


async def run_generation_and_judge(eval_set, hits_by_id, generator, judge_client):
    """Chạy song song EVAL_CONCURRENCY câu. Thời gian gần như toàn bộ là chờ
    network, nên đây là chỗ tiết kiệm lớn nhất."""
    sem = asyncio.Semaphore(EVAL_CONCURRENCY)
    counters = {"done": 0, "total": len(eval_set), "fail": 0}

    tasks = [_process_row(row, hits_by_id[row["id"]], generator, judge_client, sem, counters)
             for row in eval_set]
    try:
        rows_out = await asyncio.gather(*tasks)
    except DailyQuotaExhausted as e:
        for t in tasks:
            t.cancel() if hasattr(t, "cancel") else None
        sys.exit(
            f"\nHết quota NGÀY của Groq (TPD). Dừng để không chạy tiếp vô ích.\n"
            f"  {str(e)[:200]}\n"
            "  Chờ quota reset, hoặc dùng eval set nhỏ hơn:\n"
            '    $env:EVAL_N_STRICT="5"; $env:EVAL_N_LOOSE="5"; $env:EVAL_N_NOT_FOUND="3"'
        )

    if counters["fail"] >= 5:
        print(f"\n[warn] {counters['fail']} lời gọi judge cuối cùng đều thất bại với "
              f"{JUDGE_MODEL} — xem dòng `raw:` ở trên.")
    return list(rows_out)


FIELDNAMES = [
    "id", "category", "group", "strictness", "question", "answer", "latency_ms",
    "n_expected", "retrieved_ids", "relevance", "correctness", "faithfulness",
    "refused", "invented",
]


def save_results(rows, path=None):
    path = path or RESULTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

def print_summary(recall_groups, gen_rows):
    def avg(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (f"{sum(vals) / len(vals):.2f}", len(vals)) if vals else ("n/a", 0)

    def rate(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return "n/a"
        s, n = sum(vals), len(vals)
        lo, hi = wilson_ci(s, n)
        return f"{s / n:.0%}  [95% CI {lo:.0%}-{hi:.0%}]  n={n}"

    print("\n" + "=" * 68)
    print("EVAL SUMMARY")
    print(f"  generator : {GROQ_MODEL}")
    print(f"  judge     : {JUDGE_MODEL or '(dry run)'}")
    print("=" * 68)

    print("\nRETRIEVAL (recall — hit nếu bắt được BẤT KỲ doc liên quan)")
    for group in sorted(recall_groups):
        s = recall_groups[group]
        total = s["total"]
        parts = []
        for k in (1, 3):
            lo, hi = wilson_ci(s[k], total)
            parts.append(f"@{k} {s[k] / total:.0%} [{lo:.0%}-{hi:.0%}]")
        print(f"  {group:22} {'  '.join(parts)}  n={total}")

    print("\nANSWER QUALITY (1-5, judge)")
    for group in sorted({r["group"] for r in gen_rows if r["relevance"] is not None}):
        rows = [r for r in gen_rows if r["group"] == group]
        rel, n_rel = avg("relevance", rows)
        cor, _ = avg("correctness", rows)
        fai, _ = avg("faithfulness", rows)
        print(f"  {group:22} relevance {rel}  correctness {cor}  "
              f"faithfulness {fai}  scored={n_rel}/{len(rows)}")

    print("\nREFUSAL / HALLUCINATION")
    for group in ("product_not_found", "out_of_scope"):
        rows = [r for r in gen_rows if r["group"] == group]
        if not rows:
            continue
        print(f"  {group:22} refused  {rate('refused', rows)}")
        print(f"  {group:22} invented {rate('invented', rows)}")

    lat = [r["latency_ms"] for r in gen_rows if r["latency_ms"] is not None]
    if lat:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))]
        print(f"\nLATENCY  p50 {p50:.0f} ms   p95 {p95:.0f} ms   n={len(lat)}")

    print("=" * 68)
    print(f"Chi tiết: {RESULTS_PATH}")


# ------------------------------------------------------------------ dry run
class _StubRetriever:
    """Chỉ để smoke-test harness: trả về đúng doc liên quan cho câu chẵn, doc
    rác cho câu lẻ. Không phản ánh chất lượng retrieval thật."""

    def __init__(self, eval_set):
        self._by_q = {r["question"]: r for r in eval_set}

    def search(self, query, top_k=5):
        row = self._by_q.get(query, {})
        ids = [x for x in (row.get("expected_relevant_ids") or "").split(";") if x]
        good = sum(ord(c) for c in query) % 2 == 0
        out = []
        for i in range(top_k):
            if good and i < len(ids):
                doc_type = row.get("expected_doc_type", "product")
                meta = {"doc_type": doc_type, "doc_id": ids[i], "source_file": ids[i]}
            else:
                meta = {"doc_type": "product", "doc_id": f"noise{i}"}
            out.append({"id": f"stub{i}", "content": f"stub chunk {i}", "metadata": meta})
        return out


class _StubGenerator:
    async def generate(self, query, context, language="vi"):
        return {"answer": f"[stub answer for: {query[:40]}]", "latency_ms": 1.0}


class _StubJudge:
    class chat:
        class completions:
            @staticmethod
            async def create(**kwargs):
                content = kwargs["messages"][0]["content"]
                body = ('{"refused": 1, "invented": 0}' if "declined" in content
                        else '{"relevance": 3, "correctness": 3, "faithfulness": 4}')

                class R:
                    choices = [type("C", (), {"message": type("M", (), {"content": body})})]
                return R()


async def main_async():
    set_seed()
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval questions from {EVAL_SET_PATH}")

    if DRY_RUN:
        print("[dry run] dùng stub retriever/generator/judge — không gọi Groq\n")
        retriever, generator, judge_client = (
            _StubRetriever(eval_set), _StubGenerator(), _StubJudge(),
        )
    else:
        from src.rag.retriever import Retriever

        retriever = Retriever(persist_dir=CHROMA_PERSIST_DIR,
                              embedding_model=EMBEDDING_MODEL, top_k=TOP_K)

        if RECALL_ONLY:
            # Không cần Groq client, không cần cả GROQ_API_KEY.
            generator = judge_client = None
        else:
            from groq import AsyncGroq
            from src.rag.generator import Generator

            generator = Generator()
            judge_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    print(f"Retrieving (batched, top_k={TOP_K}) ...")
    hits_by_id = retrieve_all(retriever, eval_set, top_k=TOP_K)

    print("Computing retrieval recall per group ...")
    recall_groups = compute_recall_at_k(hits_by_id, eval_set, k_values=(1, 3))
    if RECALL_ONLY:
        print("[recall only] bỏ qua generation + judge — không gọi Groq")
        rows = [{
            **{k: "" for k in FIELDNAMES},
            "id": r["id"],
            "category": r["category"],
            "group": _group_of(r),
            "strictness": r.get("strictness", ""),
            "question": r["question"],
            "n_expected": len(_parse_relevant_ids(r["expected_relevant_ids"])),
            "retrieved_ids": ";".join(
                h["metadata"].get("doc_id", h["id"]) for h in hits_by_id[r["id"]]),
        } for r in eval_set]
        save_results(rows)
        print_summary(recall_groups, [])
        return
    
    print(f"Running generation + judge (concurrency={EVAL_CONCURRENCY}) ...")
    gen_rows = await run_generation_and_judge(eval_set, hits_by_id, generator, judge_client)

    save_results(gen_rows)
    print_summary(recall_groups, gen_rows)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()