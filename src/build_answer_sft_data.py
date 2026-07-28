import argparse
import json
import os

import pandas as pd

from generate_answer import build_prompt, fallback_answer, load_jsonl


def main():
    ap = argparse.ArgumentParser(description="Stage 5 extra credit: build QLoRA SFT seed data")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--debug_raw", default=None)
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--target_source", choices=["existing_only", "existing_or_fallback"],
                     default="existing_or_fallback")
    ap.add_argument("--out", default="data/answer_sft_seed.jsonl")
    args = ap.parse_args()

    catalog = pd.read_parquet(args.catalog).set_index("product_id")
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    reranked = {r["query_id"]: r for r in load_jsonl(args.reranked)}
    answers = {a["query_id"]: a for a in load_jsonl(args.answers)}

    used_fallback = {}
    if args.debug_raw and os.path.exists(args.debug_raw):
        for rec in load_jsonl(args.debug_raw):
            used_fallback[rec["query_id"]] = bool(rec.get("used_fallback"))

    rows = []
    skipped_fallback = 0
    for qid, answer in answers.items():
        if qid not in queries or qid not in reranked:
            continue
        is_fallback = used_fallback.get(qid, False)
        if is_fallback and args.target_source == "existing_only":
            skipped_fallback += 1
            continue

        q = queries[qid]
        hits = sorted(reranked[qid].get("results", []), key=lambda h: h["rank"])[: args.top_k]
        products = [
            {"product_id": h["product_id"], **catalog.loc[h["product_id"]].to_dict()}
            for h in hits if h["product_id"] in catalog.index
        ]
        if not products:
            continue
        prompt = build_prompt(q, products)

        target = dict(answer)
        target.pop("query_id", None)
        rows.append({
            "query_id": qid,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
            ],
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} SFT examples to {args.out} "
          f"(skipped {skipped_fallback} fallback answers)")


if __name__ == "__main__":
    main()
