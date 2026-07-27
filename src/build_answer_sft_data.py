"""Stage 5 extra credit — build a small SFT seed dataset for QLoRA fine-tuning.

Turns a stronger local model's accepted Stage 5 answers (see generate_answer.py)
into (prompt, target JSON) training pairs for a weaker/base model, so the
weaker model can be taught the same prompt -> valid-JSON behavior via LoRA.

The exact prompt reconstructed here is the same one generate_answer.py would
build for that query (same catalog/queries/reranked inputs), so the SFT
example is faithful to what the model actually saw.

Usage:
    uv run python src/build_answer_sft_data.py \
        --answers outputs/final_answers.local_qwen3b.jsonl \
        --debug_raw reports/raw_llm_debug.local_qwen3b.jsonl \
        --catalog data/catalog_subset.parquet \
        --queries data/queries.jsonl \
        --reranked outputs/reranked_results.jsonl \
        --top_k 8 \
        --target_source existing_or_fallback \
        --out data/answer_sft_seed.local_qwen3b.jsonl
"""

import argparse
import json
import os

import pandas as pd

from generate_answer import build_prompt, fallback_answer, load_jsonl


def main():
    ap = argparse.ArgumentParser(description="Stage 5 extra credit: build QLoRA SFT seed data")
    ap.add_argument("--answers", required=True,
                     help="Stage 5 final_answers.jsonl produced by a (stronger) model run")
    ap.add_argument("--debug_raw", default=None,
                     help="Matching --debug_raw_out log from generate_answer.py (used to skip "
                          "fallback answers unless --target_source includes them)")
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--target_source", choices=["existing_only", "existing_or_fallback"],
                     default="existing_or_fallback",
                     help="existing_only: skip queries whose answer used the deterministic "
                          "fallback. existing_or_fallback: use the fallback JSON as the training "
                          "target too, so every query contributes an example.")
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
