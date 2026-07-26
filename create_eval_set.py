import csv

rows = [r for r in csv.DictReader(open("data/eval/eval_set.csv", encoding="utf-8"))
        if r["category"] in ("policy", "out_of_scope")]
fn = ["id", "question", "language", "category",
      "expected_doc_type", "expected_relevant_ids", "expected_answer_keypoints"]

with open("data/eval/eval_set_manual.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fn})

print(len(rows), "dòng manual")   # -> 30