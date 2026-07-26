import csv

rows = list(csv.DictReader(open("data/eval/results_recall.csv", encoding="utf-8")))
pol = [r for r in rows if r["group"] == "policy"]
print(f"{len(pol)} câu policy\n")

miss = 0
for r in pol:
    ids = [x for x in r["retrieved_ids"].split(";") if x]
    # chunk policy có doc_id dạng "policy_<file>.md_<i>", product thì là số
    n_pol = sum(1 for i in ids if i.startswith("policy_"))
    if n_pol == 0:
        miss += 1
    print(f"{r['id']}  policy-chunk trong top-5: {n_pol}/5  | {r['question'][:52]}")
    if n_pol == 0:
        print(f"       -> toàn sản phẩm: {ids[:3]}")

print(f"\n{miss}/{len(pol)} câu policy KHÔNG có một chunk policy nào trong top-5")