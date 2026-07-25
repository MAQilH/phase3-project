"""Stage 3 — cross-encoder / multimodal reranker fusion.

Reranks the Stage 2 candidate pool with a multimodal cross-encoder that
scores (query, product) pairs, where either side may carry text, image, or
both. Produces two ablations selectable with --strategy:

  ce_only   final_score = cross_encoder_score
  fused     final_score = w_ce*z(ce) + w_dense*z(dense) + w_sparse*z(sparse)

Usage:
    uv run python src/rerank.py \
        --catalog data/catalog_subset.parquet \
        --queries data/queries.jsonl \
        --retrieval outputs/retrieval_results.jsonl \
        --reranker jinaai/jina-reranker-m0 \
        --rerank_top_n 100 \
        --strategy fused \
        --out outputs/reranked_results.jsonl
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

FUSION_WEIGHTS = {
    "text": {"w_ce": 0.65, "w_dense": 0.20, "w_sparse": 0.15},
    "image": {"w_ce": 0.60, "w_dense": 0.40, "w_sparse": 0.00},
    "image_text": {"w_ce": 0.55, "w_dense": 0.30, "w_sparse": 0.15},
}


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def zscore(values):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=np.float64)
    if np.all(np.isnan(arr)):
        return [0.0] * len(values)
    mean = np.nanmean(arr)
    std = np.nanstd(arr) or 1.0
    arr = np.where(np.isnan(arr), mean, arr)
    return list((arr - mean) / std)


def build_product_text_for_rerank(row):
    parts = [row.get("title") or ""]
    if row.get("product_type"):
        parts.append(f"Type: {row['product_type']}")
    if row.get("category_path"):
        parts.append(f"Category: {row['category_path']}")
    if row.get("brand"):
        parts.append(f"Brand: {row['brand']}")
    if row.get("color"):
        parts.append(f"Color: {row['color']}")
    if row.get("material"):
        parts.append(f"Material: {row['material']}")
    if row.get("style"):
        parts.append(f"Style: {row['style']}")
    if row.get("bullet_points"):
        parts.append(row["bullet_points"][:300])
    if row.get("description"):
        parts.append(row["description"][:200])
    return " | ".join(p for p in parts if p)


def build_rerank_query(q):
    text, image = q.get("query_text"), q.get("query_image_path")
    if text and image:
        return {"text": text, "image": image}
    if image:
        return image
    return text


def main():
    ap = argparse.ArgumentParser(description="Stage 3: cross-encoder fusion and reranking")
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranker", default="jinaai/jina-reranker-m0")
    ap.add_argument("--rerank_top_n", type=int, default=100)
    ap.add_argument("--strategy", choices=["ce_only", "fused"], default="fused")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/reranked_results.jsonl")
    ap.add_argument("--report", default="reports/reranking_summary.md")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(args.reranker, trust_remote_code=True, device=device)

    catalog = pd.read_parquet(args.catalog).set_index("product_id")
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    retrieval = load_jsonl(args.retrieval)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    examples = []
    negation_example = None
    image_text_example = None

    with open(args.out, "w", encoding="utf-8") as out_fh:
        for rec in retrieval:
            q = queries[rec["query_id"]]
            hits = rec["results"][: args.rerank_top_n]
            if not hits:
                out_fh.write(json.dumps({
                    "query_id": rec["query_id"],
                    "run_name": f"hybrid_cross_encoder_{args.strategy}",
                    "reranker_type": "multimodal_cross_encoder",
                    "results": [],
                }) + "\n")
                continue

            before_ids = [h["product_id"] for h in hits]

            rerank_query = build_rerank_query(q)
            products = [{
                "text": build_product_text_for_rerank(catalog.loc[h["product_id"]].to_dict()),
                "image": catalog.loc[h["product_id"]]["image_path"],
            } for h in hits]

            pairs = [(rerank_query, p) for p in products]
            ce_scores = reranker.predict(pairs)
            ce_scores = np.asarray(ce_scores, dtype=np.float64).ravel()

            for h, ce in zip(hits, ce_scores):
                h["cross_encoder_score"] = float(ce)

            if args.strategy == "ce_only":
                for h in hits:
                    h["final_score"] = h["cross_encoder_score"]
            else:
                weights = FUSION_WEIGHTS[q["query_type"]]
                ce_z = zscore([h["cross_encoder_score"] for h in hits])
                dense_z = zscore([h.get("dense_score") for h in hits])
                sparse_z = zscore([h.get("sparse_score") for h in hits])
                for h, cez, dz, sz in zip(hits, ce_z, dense_z, sparse_z):
                    h["final_score"] = (
                        weights["w_ce"] * cez
                        + weights["w_dense"] * dz
                        + weights["w_sparse"] * sz
                    )

            hits.sort(key=lambda h: -h["final_score"])
            for rank, h in enumerate(hits, start=1):
                h["rank"] = rank

            record = {
                "query_id": rec["query_id"],
                "run_name": f"hybrid_cross_encoder_{args.strategy}",
                "reranker_type": "multimodal_cross_encoder",
                "results": hits,
            }
            out_fh.write(json.dumps(record) + "\n")

            if len(examples) < 3:
                examples.append((q, before_ids[:5], hits[:5]))
            if negation_example is None and any(
                kw in (q.get("query_text") or "").lower()
                for kw in ["not ", "without", "no leather"]
            ):
                negation_example = (q, hits[:5])
            if image_text_example is None and q["query_type"] == "image_text":
                image_text_example = (q, hits[:5])

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("# Stage 3 — Reranking Summary\n\n")
        fh.write(f"- Reranker model: `{args.reranker}` (multimodal cross-encoder)\n")
        fh.write("- Reranker input: `(rerank_query, rerank_product)` pairs, where each side is "
                  "text, image, or a `{\"text\":.., \"image\":..}` dict\n")
        fh.write(f"- Reranking depth (rerank_top_n): {args.rerank_top_n}\n")
        fh.write("- Score normalization: per-query z-score for cross-encoder, dense, and sparse scores\n")
        fh.write(f"- Strategy: {args.strategy}"
                 + (f", fusion weights: {json.dumps(FUSION_WEIGHTS)}\n" if args.strategy == "fused" else "\n"))
        fh.write(f"- Device: {device}\n\n")

        fh.write("## Before / after examples\n\n")
        for q, before, after in examples:
            fh.write(f"**Query {q['query_id']}** ({q['query_type']}): "
                     f"{q.get('query_text') or q.get('query_image_path')}\n\n")
            fh.write("Before reranking (top 5): "
                     + ", ".join(before) + "\n\n")
            fh.write("After reranking (top 5): "
                     + ", ".join(h["product_id"] for h in after) + "\n\n")

        if negation_example:
            q, top5 = negation_example
            fh.write(f"## Negation example\n\nQuery {q['query_id']}: {q.get('query_text')}\n\n")
            fh.write("Top results after reranking: "
                     + ", ".join(h["product_id"] for h in top5) + "\n\n")
        if image_text_example:
            q, top5 = image_text_example
            fh.write(f"## Image+text example\n\nQuery {q['query_id']}: "
                     f"{q.get('query_text')} (image: {q.get('query_image_path')})\n\n")
            fh.write("Top results after reranking: "
                     + ", ".join(h["product_id"] for h in top5) + "\n\n")

    print(f"Wrote reranked results to {args.out}")


if __name__ == "__main__":
    main()
