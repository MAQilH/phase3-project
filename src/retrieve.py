"""Stage 2 — text / image / image+text candidate retrieval.

Encodes each query with the same multimodal dense model used in Stage 1,
retrieves top-K dense candidates from the FAISS index, retrieves top-K
sparse candidates from the CSR sparse index (for text-bearing queries),
merges them, and computes a prefusion score (RRF or weighted-normalized
score fusion) as the non-reranked hybrid baseline.

Usage:
    uv run python src/retrieve.py \
        --catalog data/catalog_subset.parquet \
        --queries data/queries.jsonl \
        --index_dir artifacts/index \
        --emb_dir artifacts/embeddings \
        --top_k_dense 100 \
        --top_k_sparse 100 \
        --fusion rrf \
        --out outputs/retrieval_results.jsonl
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_queries(path):
    queries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_sparse_query_encoder(sparse_model_name, device):
    from build_index import SpladeEncoder
    return SpladeEncoder(sparse_model_name, device)


def dense_search(index, query_vec, top_k):
    scores, idx = index.search(np.ascontiguousarray([query_vec], dtype=np.float32), top_k)
    return list(zip(idx[0].tolist(), scores[0].tolist()))


def sparse_search(sparse_matrix, query_indices, query_values, vocab_size, top_k):
    q = np.zeros(vocab_size, dtype=np.float32)
    q[query_indices] = query_values
    scores = np.asarray(sparse_matrix.dot(q)).ravel()
    top_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


def normalize_scores(pairs):
    """Min-max normalize a list of (id, score) pairs to [0, 1]."""
    if not pairs:
        return {}
    scores = np.array([s for _, s in pairs], dtype=np.float32)
    lo, hi = scores.min(), scores.max()
    span = (hi - lo) or 1.0
    return {i: float((s - lo) / span) for i, s in pairs}


def rrf_fuse(rank_lists, k=60):
    """rank_lists: list of ordered id lists (best first). Returns {id: rrf_score}."""
    scores = {}
    for ranks in rank_lists:
        for rank, item_id in enumerate(ranks, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_fuse(dense_norm, sparse_norm, w_dense, w_sparse):
    ids = set(dense_norm) | set(sparse_norm)
    return {
        i: w_dense * dense_norm.get(i, 0.0) + w_sparse * sparse_norm.get(i, 0.0)
        for i in ids
    }


def build_result_list(product_ids, dense_pairs, sparse_pairs, fusion, rrf_k, weights, max_pool):
    dense_map = dict(dense_pairs)
    sparse_map = dict(sparse_pairs)
    dense_ranked_ids = [i for i, _ in dense_pairs]
    sparse_ranked_ids = [i for i, _ in sparse_pairs]

    if fusion == "rrf":
        rank_lists = [dense_ranked_ids]
        if sparse_pairs:
            rank_lists.append(sparse_ranked_ids)
        fused = rrf_fuse(rank_lists, k=rrf_k)
    else:
        dense_norm = normalize_scores(dense_pairs)
        sparse_norm = normalize_scores(sparse_pairs) if sparse_pairs else {}
        w_dense, w_sparse = weights
        if not sparse_pairs:
            w_dense, w_sparse = 1.0, 0.0
        fused = weighted_fuse(dense_norm, sparse_norm, w_dense, w_sparse)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:max_pool]

    results = []
    for rank, (row_idx, fused_score) in enumerate(ordered, start=1):
        source = []
        if row_idx in dense_map:
            source.append("dense")
        if row_idx in sparse_map:
            source.append("sparse")
        results.append({
            "rank": rank,
            "product_id": product_ids[row_idx],
            "dense_score": dense_map.get(row_idx),
            "sparse_score": sparse_map.get(row_idx),
            "prefusion_score": fused_score,
            "source": source,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description="Stage 2: hybrid candidate retrieval")
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--emb_dir", default="artifacts/embeddings")
    ap.add_argument("--top_k_dense", type=int, default=100)
    ap.add_argument("--top_k_sparse", type=int, default=100)
    ap.add_argument("--max_pool", type=int, default=200)
    ap.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--w_dense", type=float, default=0.55)
    ap.add_argument("--w_sparse", type=float, default=0.45)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--report", default="reports/retrieval_summary.md")
    args = ap.parse_args()

    import faiss
    from scipy import sparse as sps
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")

    catalog = pd.read_parquet(args.catalog)
    product_ids = catalog["product_id"].tolist()

    manifest = json.load(open(os.path.join(args.index_dir, "index_manifest.json")))
    dense_index = faiss.read_index(os.path.join(args.index_dir, "dense_index.faiss"))
    sparse_matrix = sps.load_npz(os.path.join(args.index_dir, "sparse_index.npz"))
    vocab_size = manifest["sparse_vocab_size"]

    from build_index import load_dense_model, encode_query_dense
    dense_model = load_dense_model(manifest["dense_model"], device)
    sparse_encoder = load_sparse_query_encoder(manifest["sparse_model"], device)

    queries = load_queries(args.queries)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n_by_type = {"text": 0, "image": 0, "image_text": 0}
    pool_sizes = []
    both_counts = []
    examples = []

    with open(args.out, "w", encoding="utf-8") as out_fh:
        for q in queries:
            qtype = q["query_type"]
            n_by_type[qtype] = n_by_type.get(qtype, 0) + 1
            sparse_applicable = qtype != "image"

            query_vec = encode_query_dense(dense_model, q.get("query_text"), q.get("query_image_path"))
            dense_pairs = dense_search(dense_index, query_vec, args.top_k_dense)

            sparse_pairs = []
            if sparse_applicable and q.get("query_text"):
                sp_idx, sp_val = sparse_encoder.encode_query(q["query_text"])
                sparse_pairs = sparse_search(sparse_matrix, sp_idx, sp_val, vocab_size, args.top_k_sparse)

            results = build_result_list(
                product_ids, dense_pairs, sparse_pairs, args.fusion, args.rrf_k,
                (args.w_dense, args.w_sparse), args.max_pool,
            )
            for r in results:
                if not sparse_applicable:
                    r["sparse_score"] = None

            record = {
                "query_id": q["query_id"],
                "query_type": qtype,
                "run_name": "hybrid_dense_sparse_prefusion",
                "sparse_applicable": sparse_applicable,
                "results": results,
            }
            out_fh.write(json.dumps(record) + "\n")

            pool_sizes.append(len(results))
            both_counts.append(sum(1 for r in results if len(r["source"]) == 2))
            if len(examples) < 3:
                examples.append((q, results[:5]))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("# Stage 2 — Retrieval Summary\n\n")
        fh.write(f"- Queries by type: {n_by_type}\n")
        fh.write(f"- Dense top-K: {args.top_k_dense}, Sparse top-K: {args.top_k_sparse}\n")
        fh.write(f"- Prefusion method: {args.fusion}"
                 + (f" (k={args.rrf_k})\n" if args.fusion == "rrf"
                    else f" (w_dense={args.w_dense}, w_sparse={args.w_sparse})\n"))
        fh.write(f"- Average unique candidates per query: {np.mean(pool_sizes):.1f}\n")
        fh.write(f"- Average candidates found by both dense and sparse: {np.mean(both_counts):.1f}\n\n")
        fh.write("## Example top-5 results\n\n")
        for q, top5 in examples:
            fh.write(f"**Query {q['query_id']}** ({q['query_type']}): "
                     f"{q.get('query_text') or q.get('query_image_path')}\n\n")
            for r in top5:
                fh.write(f"- rank {r['rank']}: {r['product_id']} "
                         f"(dense={r['dense_score']}, sparse={r['sparse_score']}, "
                         f"prefusion={r['prefusion_score']:.4f}, source={r['source']})\n")
            fh.write("\n")

    print(f"Wrote retrieval results for {len(queries)} queries to {args.out}")


if __name__ == "__main__":
    main()
