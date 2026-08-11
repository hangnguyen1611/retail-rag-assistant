import asyncio
import csv
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.backend import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    EVAL_RESULTS_PATH,
    EVAL_SET_PATH,
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

# Số câu xử lý song song
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "1"))

# --- Judge model: phải khai báo tường minh và phải KHÁC model sinh câu trả lời ---
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

# Regex chỉ còn dùng làm fallback khi judge parse lỗi
REFUSAL_PATTERNS = [
    "không có thông tin", "không tìm thấy", "không có sản phẩm", "không có mẫu",
    "hiện không có", "chưa có", "liên hệ nhân viên", "không thể trả lời", "không nằm trong",
    "does not contain", "don't have this information", "do not have this information",
    "i don't have information", "we don't carry", "not available", "no matching",
    "contact support", "contact our support", "i'm not able to answer", "outside the scope",
]


def is_refusal(answer):
    """
    Kiểm tra câu trả lời có phải là một lời từ chối hay không, dựa trên danh sách pattern regex/từ khoá cố định (REFUSAL_PATTERNS).
    Chỉ dùng làm fallback khi judge (LLM) parse JSON lỗi — không phải cách chấm chính, vì cách match keyword rất dễ bỏ sót các câu 
    từ chối diễn đạt khác đi.
    """
    if not answer:
        return False
    text = answer.lower()
    return any(p in text for p in REFUSAL_PATTERNS)


def wilson_ci(successes, n, z=1.96):
    """
    Tính khoảng tin cậy 95% (mặc định z=1.96) cho một tỷ lệ, dùng công thức Wilson score interval — chính xác hơn khoảng tin cậy chuẩn 
    (normal approximation) khi n nhỏ hoặc p gần 0/1, vốn là tình huống thường gặp với eval set nhỏ.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_eval_set(path=EVAL_SET_PATH):
    """Đọc file eval set (CSV) thành list các dict, mỗi dict là một câu hỏi kèm metadata (category, expected_relevant_ids, ...)"""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_relevant_ids(raw):
    """Parse chuỗi id phân tách bằng dấu ';' (từ cột expected_relevant_ids) thành một set các id, bỏ khoảng trắng thừa và phần tử rỗng"""
    if not raw:
        return set()
    return {x.strip() for x in raw.split(";") if x.strip()}


def _is_hit(hit, doc_type, expected_ids):
    """
    Kiểm tra một kết quả retrieve (hit) có khớp với đáp án mong đợi (expected_ids) hay không, tuỳ theo loại tài liệu.
    - product: so khớp theo metadata['doc_id']
    - policy: so khớp theo metadata['source_file']
    (Hai loại dùng field định danh khác nhau nên phải tách logic.)
    """
    metadata = hit.get("metadata", {})
    if doc_type == "product":
        return metadata.get("doc_type") == "product" and metadata.get("doc_id") in expected_ids
    if doc_type == "policy":
        return metadata.get("doc_type") == "policy" and metadata.get("source_file") in expected_ids
    return False


def _group_of(row):
    """
    Xác định nhãn nhóm (group) để báo cáo riêng cho một row eval set.
    Với category "product", tách thêm theo strictness (strict/loose) vì hai loại câu hỏi này có độ khó và cách chấm khác hẳn nhau, 
    gộp chung sẽ làm mờ số liệu.
    """
    if row["category"] == "product":
        return f"product/{row.get('strictness') or 'loose'}"
    return row["category"]


def retrieve_all(retriever, eval_set, top_k):
    """
    Chạy retrieval một lần cho toàn bộ câu hỏi trong eval set, dùng chung kết quả cho cả bước tính hit lẫn bước generation 
    (tránh gọi retriever 2 lần cho cùng 1 câu).
    Ưu tiên gọi batch (retriever.search_many) nếu retriever hỗ trợ, để nhanh hơn nếu không thì fallback gọi search() từng câu một.
    """
    questions = [r["question"] for r in eval_set]
    if hasattr(retriever, "search_many"):
        all_hits = retriever.search_many(questions, top_k=top_k)
    else:
        all_hits = [retriever.search(q, top_k=top_k) for q in questions]
    return {r["id"]: h for r, h in zip(eval_set, all_hits)}


def compute_hit_at_k(hits_by_id, eval_set, k_values=(1, 3)):
    """
    Tính Hit@k (recall) cho từng nhóm câu hỏi, dựa trên kết quả retrieve đã có sẵn (không gọi lại retriever).
    Chỉ tính cho category "product" và "policy" — các category khác không có "đáp án đúng" để retrieve nên bỏ qua.
    """
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


# ------------------------------- Judges ----------------------------------- 
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
    """
    Trích xuất và parse object JSON đầu tiên từ output thô của judge model, xử lý các trường hợp lệch chuẩn thường gặp: 
    - Có thẻ <think>...</think> (reasoning model)
    - Có markdown code fence ```json
    - Có text thừa quanh JSON
    """
    if not raw or not raw.strip():
        raise ValueError("judge trả về content RỖNG (model reasoning? thử reasoning_effort='none')")
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"không tìm thấy JSON trong: {text[:200]!r}")
    return json.loads(m.group(0))


MAX_RETRIES = 5


class DailyQuotaExhausted(RuntimeError):
    """Hết quota NGÀY (TPD/RPD) — retry vô nghĩa, phải dừng cả run."""


def _parse_retry_after(msg):
    """Parse thời gian cần chờ (tính bằng s) từ thông báo lỗi rate-limit của Groq, dạng "...try again in 1m2.5s" hoặc "...try again in 500ms"""
    m = re.search(r"try again in\s+(?:(\d+)m(?=[\d.]))?\s*([\d.]+)\s*(ms|s)\b", msg)
    if not m:
        return None
    val = float(m.group(2))
    if m.group(3) == "ms":
        val /= 1000.0
    return int(m.group(1) or 0) * 60 + val


async def _with_retry(coro_factory, label):
    """
    Wrapper retry cho các lệnh gọi API (generation hoặc judge), phân biệt 2 loại lỗi:
    - Rate limit NGÀY (TPD/RPD): không có ý nghĩa để retry trong cùng lần chạy -> raise DailyQuotaExhausted ngay để dừng cả run.
    - Rate limit TẠM THỜI (TPM/RPM/503/over capacity): retry tối đa MAX_RETRIES lần, chờ đúng thời gian Groq yêu cầu (qua
    _parse_retry_after) nếu có, không thì dùng backoff số mũ 2^attempt, chặn trần 90s.
    - Lỗi khác: raise ngay, không retry.
    """
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
    """
    Gọi judge model qua Groq API với 1 prompt, cấu hình temperature=0 và ép response dạng JSON object. 
    Với model qwen3 (mặc định bật reasoning ngầm, tốn quota ngày dù content trả về rỗng), tắt reasoning bằng reasoning_effort="none
    """
    kwargs = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    if "qwen3" in JUDGE_MODEL.lower():
        kwargs["reasoning_effort"] = "none"

    async def call():
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    return await _with_retry(call, "judge")


async def judge_quality(client, row, answer):
    """
    Chấm 3 tiêu chí relevance/correctness/faithfulness (thang 1-5) cho một câu trả lời, dùng prompt JUDGE_STRICT (câu hỏi có 1 đáp án đúng
    duy nhất) hoặc JUDGE_LOOSE (câu hỏi có nhiều sản phẩm cùng thoả mãn) tuỳ theo row["strictness"].
    """
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
    """
    Chấm xem chatbot có từ chối đúng cách hay không, cho các câu thuộc category out_of_scope / product_not_found. Không dùng regex trực tiếp
    (is_refusal) làm cách chấm chính, vì trợ lý có thể từ chối bằng câu văn không khớp pattern nào nhưng vẫn đúng — chỉ dùng regex làm 
    fallback khi judge lỗi.
    """
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


# ------------------------------- Run ----------------------------------- 
async def _process_row(row, hits, generator, judge_client, sem, counters):
    """
    Xử lý đầy đủ một câu hỏi: build context từ hits đã retrieve sẵn, gọi generator sinh câu trả lời, rồi gọi judge phù hợp 
    (judge_refusal hoặc judge_quality tuỳ category), cuối cùng gói kết quả thành 1 row output. 
    Dùng semaphore (sem) để giới hạn số câu chạy song song
    """
    async with sem:
        qid, category = row["id"], row["category"]
        context = "\n\n---\n\n".join(h["content"] for h in hits)

        gen_failed = False
        try:
            gen = await _with_retry(
                lambda: generator.generate(row["question"], context, language=row.get("language") or "auto"),
                "gen",
            )
            answer, latency_ms = gen["answer"], gen["latency_ms"]
        except DailyQuotaExhausted:
            raise
        except Exception as e:
            print(f"  [generation error] {qid}: {e}")
            answer, latency_ms = "", None
            gen_failed = True

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
            "gen_failed": gen_failed,
        }

        # Không chấm judge cho câu generation đã lỗi
        if not gen_failed:
            if category in ("out_of_scope", "product_not_found"):
                out.update(await judge_refusal(judge_client, row, answer))
            else:
                out.update(await judge_quality(judge_client, row, answer))

        if not gen_failed and out["relevance"] is None and out["invented"] is None:
            counters["fail"] += 1
        else:
            counters["fail"] = 0
        counters["done"] += 1
        tag = " [GEN FAILED]" if gen_failed else ""
        print(f"  done {counters['done']}/{counters['total']} {qid} ({out['group']}){tag}")
        return out


async def run_generation_and_judge(eval_set, hits_by_id, generator, judge_client):
    """
    Chạy generation + judge song song cho toàn bộ eval_set với concurrency giới hạn bởi EVAL_CONCURRENCY. 
    Dùng asyncio.gather(..., return_exceptions=True) thay vì để lỗi raise thẳng, để KHÔNG mất các câu đã chấm xong thành công
    khi một câu bất kỳ gặp DailyQuotaExhausted giữa chừng — nếu không, một lần hết quota giữa chừng sẽ làm mất toàn bộ kết quả 
    của batch, phải chạy lại từ đầu và tốn quota lại.
    """
    sem = asyncio.Semaphore(EVAL_CONCURRENCY)
    counters = {"done": 0, "total": len(eval_set), "fail": 0}

    tasks = [_process_row(row, hits_by_id[row["id"]], generator, judge_client, sem, counters)
             for row in eval_set]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rows_out = []
    quota_msg = None
    other_errors = 0
    for r in results:
        if isinstance(r, DailyQuotaExhausted):
            quota_msg = quota_msg or str(r)
        elif isinstance(r, Exception):
            other_errors += 1
            print(f"  [unexpected error] {r!r}")
        else:
            rows_out.append(r)

    if counters["fail"] >= 5:
        print(f"\n[warn] {counters['fail']} lời gọi judge cuối cùng đều thất bại với "
              f"{JUDGE_MODEL} — xem dòng `raw:` ở trên.")
    if other_errors:
        print(f"[warn] {other_errors} câu lỗi không xác định, bị bỏ qua (không tính vào kết quả).")

    return rows_out, quota_msg


FIELDNAMES = [
    "id", "category", "group", "strictness", "question", "answer", "latency_ms",
    "n_expected", "retrieved_ids", "relevance", "correctness", "faithfulness",
    "refused", "invented", "gen_failed",
]


def save_results(rows, path=None):
    """
    Ghi danh sách kết quả ra file CSV theo đúng FIELDNAMES, tạo thư mục cha nếu chưa tồn tại. 
    Ghi đè toàn bộ file (không append).
    """
    path = path or RESULTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


_NUMERIC_FIELDS = ("latency_ms", "relevance", "correctness", "faithfulness", "refused", "invented")


def load_existing_results(path=None):
    """
    Đọc kết quả đã chấm từ lần chạy trước (nếu file tồn tại), phục vụ cơ chế resume.
    Chỉ chạy tiếp các câu chưa có kết quả thay vì chạy lại từ đầu và tốn lại token/quota cho các câu đã xong.
    Chỉ giữ lại các row có giá trị 'answer' (tức đã thực sự gọi Groq thành công), bỏ qua row rỗng (vd từ recall-only run). 
    Convert các field số về float/None.
    """
    path = path or RESULTS_PATH
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("answer"):
                continue
            for k in _NUMERIC_FIELDS:
                v = row.get(k)
                row[k] = float(v) if v not in (None, "") else None
            out[row["id"]] = row
    return out

def print_summary(recall_groups, gen_rows):
    """
    In báo cáo tổng kết ra console: recall theo nhóm (Hit@1, Hit@3 kèm Wilson CI), chất lượng câu trả lời trung bình theo nhóm 
    (relevance/correctness/faithfulness), tỷ lệ refuse/invented cho nhóm refusal (kèm Wilson CI), và latency p50/p95.
    """
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

    n_failed = sum(1 for r in gen_rows if r.get("gen_failed"))
    if n_failed:
        print(f"\n[!] {n_failed}/{len(gen_rows)} câu generation lỗi (rate limit/timeout) "
              "-- bị loại khỏi thống kê chất lượng bên dưới, không tính là judge chấm thấp.")

    scoreable = [r for r in gen_rows if not r.get("gen_failed")]

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
    for group in sorted({r["group"] for r in scoreable if r["relevance"] is not None}):
        rows = [r for r in scoreable if r["group"] == group]
        rel, n_rel = avg("relevance", rows)
        cor, _ = avg("correctness", rows)
        fai, _ = avg("faithfulness", rows)
        print(f"  {group:22} relevance {rel}  correctness {cor}  "
              f"faithfulness {fai}  scored={n_rel}/{len(rows)}")

    print("\nREFUSAL / HALLUCINATION")
    for group in ("product_not_found", "out_of_scope"):
        rows = [r for r in scoreable if r["group"] == group]
        if not rows:
            continue
        print(f"  {group:22} refused  {rate('refused', rows)}")
        print(f"  {group:22} invented {rate('invented', rows)}")

    lat = [r["latency_ms"] for r in scoreable if r["latency_ms"] is not None]
    if lat:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))]
        print(f"\nLATENCY  p50 {p50:.0f} ms   p95 {p95:.0f} ms   n={len(lat)}")

    print("=" * 68)
    print(f"Chi tiết: {RESULTS_PATH}")


# -------------------------------- Dry run ----------------------------------
class _StubRetriever:
    """
    Retriever giả lập chỉ để smoke-test harness (không phản ánh chất lượng retrieval thật). 
    Với câu hỏi có tổng mã ký tự chẵn, trả về đúng doc mong đợi; câu lẻ thì trả về toàn doc rác — mục đích để kiểm tra
    logic tính Hit@k có chạy đúng hay không, không phải để đánh giá model.
    """

    def __init__(self, eval_set):
        self._by_q = {r["question"]: r for r in eval_set}

    def search(self, query, top_k=5):
        """Trả về top_k hit giả cho một câu hỏi"""
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
    """Sinh câu trả lời giả (echo lại 40 ký tự đầu của câu hỏi), dùng khi DRY_RUN=1 để test luồng chạy mà không gọi Groq thật."""
    async def generate(self, query, context, language="vi"):
        return {"answer": f"[stub answer for: {query[:40]}]", "latency_ms": 1.0}


class _StubJudge:
    """
    Judge giả: nếu prompt có chữ 'declined' (JUDGE_REFUSAL) trả về JSON refused/invented cố định.
    Ngược lại trả JSON relevance/correctness/faithfulness cố định. Chỉ dùng cho DRY_RUN.
    """
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
    """
    Luồng xử lý:
    - Load eval set.
    - Khởi tạo retriever/generator/judge_client — dùng stub nếu DRY_RUN=1, dùng object thật (Retriever/Generator/AsyncGroq) 
    nếu không, bỏ qua generator/judge nếu RECALL_ONLY=1.
    - Retrieve toàn bộ câu hỏi một lần (retrieve_all), tính Hit@k (compute_hit_at_k).
    - Nếu RECALL_ONLY: lưu kết quả rỗng phần generation, in summary, dừng (không gọi Groq generation/judge).
    - Nếu không: load kết quả cũ (nếu EVAL_RESUME=1, mặc định bật) để chỉ chạy tiếp các câu chưa có kết quả, chạy generation+judge
    (run_generation_and_judge), gộp với kết quả cũ, sắp lại theo thứ tự eval set gốc, lưu file, in summary.
    - Nếu gặp DailyQuotaExhausted giữa chừng: vẫn lưu kết quả đã có, rồi sys.exit với thông báo hướng dẫn chạy lại (script sẽ tự resume).
    """
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

        retriever = Retriever(persist_dir=CHROMA_PERSIST_DIR, embedding_model=EMBEDDING_MODEL, top_k=TOP_K)

        if RECALL_ONLY:
            # Không cần Groq client, không cần cả GROQ_API_KEY
            generator = judge_client = None
        else:
            from groq import AsyncGroq
            from src.rag.generator import Generator

            generator = Generator()
            judge_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    print(f"Retrieving (batched, top_k={TOP_K}) ...")
    hits_by_id = retrieve_all(retriever, eval_set, top_k=TOP_K)

    print("Computing retrieval hit per group ...")
    recall_groups = compute_hit_at_k(hits_by_id, eval_set, k_values=(1, 3))
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
            "retrieved_ids": ";".join(h["metadata"].get("doc_id", h["id"]) for h in hits_by_id[r["id"]]),
        } for r in eval_set]
        save_results(rows)
        print_summary(recall_groups, [])
        return
    
    resume = os.getenv("EVAL_RESUME", "1") == "1"
    existing = load_existing_results() if resume else {}
    remaining = [r for r in eval_set if r["id"] not in existing]

    if existing:
        print(f"[resume] {len(existing)}/{len(eval_set)} câu đã có kết quả từ lần chạy trước "
              f"({RESULTS_PATH}) -- bỏ qua, chỉ chạy {len(remaining)} câu còn lại.\n"
              "         (EVAL_RESUME=0 nếu muốn chạy lại từ đầu, tốn quota lại toàn bộ.)")

    print(f"Running generation + judge (concurrency={EVAL_CONCURRENCY}) ...")
    new_rows, quota_msg = await run_generation_and_judge(remaining, hits_by_id, generator, judge_client)

    # Luôn lưu -- kể cả khi hết quota giữa chừng -- để lần chạy sau (resume)
    # không phải trả lại tiền/token cho các câu đã chấm xong.
    order = {r["id"]: i for i, r in enumerate(eval_set)}
    gen_rows = list(existing.values()) + new_rows
    gen_rows.sort(key=lambda r: order.get(r["id"], 1 << 30))

    save_results(gen_rows)
    print_summary(recall_groups, gen_rows)

    if quota_msg:
        left = len(eval_set) - len(gen_rows)
        sys.exit(
            f"\nHết quota NGÀY của Groq (TPD). Đã lưu {len(gen_rows)}/{len(eval_set)} câu vào "
            f"{RESULTS_PATH} (bao gồm {len(new_rows)} câu vừa chạy).\n"
            f"  {quota_msg[:200]}\n"
            f"  Còn {left} câu chưa chấm. Chờ quota reset rồi CHẠY LẠI ĐÚNG LỆNH CŨ --\n"
            "  script sẽ tự bỏ qua các câu đã xong (EVAL_RESUME=1 mặc định) và chỉ\n"
            f"  chấm nốt {left} câu còn lại."
        )


def main():
    """Entry point đồng bộ, chỉ gọi asyncio.run(main_async())."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()