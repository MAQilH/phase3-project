import argparse
import json
import os

import pandas as pd


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_pool(args):
    catalog = pd.read_parquet(args.catalog).set_index("product_id")
    queries = load_jsonl(args.queries)
    retrieval = {r["query_id"]: r for r in load_jsonl(args.retrieval)}
    reranked = {}
    if args.reranked and os.path.exists(args.reranked):
        reranked = {r["query_id"]: r for r in load_jsonl(args.reranked)}

    rows = []
    for q in queries:
        qid = q["query_id"]
        pooled_ids = []

        ret = retrieval.get(qid, {}).get("results", [])
        dense_only = sorted(
            [h for h in ret if "dense" in h["source"]],
            key=lambda h: -(h["dense_score"] or 0),
        )
        sparse_only = sorted(
            [h for h in ret if "sparse" in h["source"]],
            key=lambda h: -(h["sparse_score"] or 0),
        )
        prefusion = sorted(ret, key=lambda h: h["rank"])
        rerank_hits = sorted(reranked.get(qid, {}).get("results", []), key=lambda h: h["rank"])

        for pool in (dense_only, sparse_only, prefusion, rerank_hits):
            for h in pool[: args.pool_depth]:
                if h["product_id"] not in pooled_ids:
                    pooled_ids.append(h["product_id"])
            if len(pooled_ids) >= args.pool_depth * 2:
                break
        pooled_ids = pooled_ids[: max(args.pool_depth, min(len(pooled_ids), 20))]

        for pid in pooled_ids:
            product = catalog.loc[pid]
            rows.append({
                "query_id": qid,
                "query_text": q.get("query_text"),
                "query_image_path": q.get("query_image_path"),
                "product_id": pid,
                "title": product["title"],
                "product_type": product["product_type"],
                "color": product["color"],
                "material": product["material"],
                "image_path": product["image_path"],
                "relevance": "",
                "reason": "",
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.pool_out) or ".", exist_ok=True)
    df.to_csv(args.pool_out, index=False)
    print(f"Wrote {len(df)} (query, product) pairs to label in {args.pool_out}")
    print("Fill in the 'relevance' (0/1/2) and 'reason' columns, then run --finalize.")


def finalize(args):
    df = pd.read_csv(args.finalize)
    missing = df["relevance"].isna() | (df["relevance"].astype(str).str.strip() == "")
    if missing.any():
        raise SystemExit(
            f"{missing.sum()} rows still have an empty 'relevance' column in "
            f"{args.finalize}. Fill them in before finalizing."
        )
    df["relevance"] = df["relevance"].astype(int)
    if not df["relevance"].isin([0, 1, 2]).all():
        raise SystemExit("relevance values must be 0, 1, or 2.")

    os.makedirs(os.path.dirname(args.qrels_out) or ".", exist_ok=True)
    with open(args.qrels_out, "w", encoding="utf-8") as fh:
        for _, r in df.iterrows():
            fh.write(json.dumps({
                "query_id": r["query_id"],
                "product_id": r["product_id"],
                "relevance": int(r["relevance"]),
                "reason": str(r.get("reason") or ""),
            }) + "\n")
    print(f"Wrote {len(df)} qrels to {args.qrels_out}")


def main():
    ap = argparse.ArgumentParser(description="Stage 4: build and finalize the qrels judgment pool")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--pool_out", default="data/qrels_pool.csv")
    ap.add_argument("--pool_depth", type=int, default=15)
    ap.add_argument("--finalize", default=None)
    ap.add_argument("--qrels_out", default="data/qrels.jsonl")
    args = ap.parse_args()

    if args.finalize:
        finalize(args)
    else:
        build_pool(args)


if __name__ == "__main__":
    main()
