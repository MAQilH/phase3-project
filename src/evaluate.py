"""Stage 4 — evaluation metrics: Precision@K, Recall@K, MAP, NDCG, judged
coverage, computed per run and per query type, plus a qualitative error
analysis comparing dense vs sparse and pre- vs post-reranking behavior.

Usage:
    uv run python src/evaluate.py \
        --queries data/queries.jsonl \
        --qrels data/qrels.jsonl \
        --runs outputs/retrieval_results.jsonl outputs/reranked_results.jsonl \
        --out reports/evaluation_metrics.csv
"""

import argparse
import json
import math
import os
from collections import defaultdict

import pandas as pd


# ===========================================================================
# Core metric functions (also imported + unit-checked by src/tests.py)
# ===========================================================================
def precision_at_k(ranked_ids, qrels, k):
    top = ranked_ids[:k]
    if not top:
        return 0.0
    relevant = sum(1 for pid in top if qrels.get(pid, 0) > 0)
    return relevant / len(top)


def recall_at_k(ranked_ids, qrels, k):
    total_relevant = sum(1 for rel in qrels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0
    top = ranked_ids[:k]
    relevant = sum(1 for pid in top if qrels.get(pid, 0) > 0)
    return relevant / total_relevant


def average_precision(ranked_ids, qrels):
    total_relevant = sum(1 for rel in qrels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precisions = []
    for i, pid in enumerate(ranked_ids, start=1):
        if qrels.get(pid, 0) > 0:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / total_relevant


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, qrels, k):
    top = ranked_ids[:k]
    gains = [qrels.get(pid, 0) for pid in top]
    dcg = _dcg(gains)
    ideal_gains = sorted(qrels.values(), reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def judged_coverage(ranked_ids, qrels, k=20):
    top = ranked_ids[:k]
    if not top:
        return 0.0
    judged = sum(1 for pid in top if pid in qrels)
    return judged / len(top)


# ===========================================================================
# I/O helpers
# ===========================================================================
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_qrels(path):
    by_query = defaultdict(dict)
    for r in load_jsonl(path):
        by_query[r["query_id"]][r["product_id"]] = r["relevance"]
    return by_query


def ranked_ids_for(record):
    hits = sorted(record["results"], key=lambda h: h["rank"])
    return [h["product_id"] for h in hits]


def reorder_by(record, key):
    hits = [h for h in record["results"] if h.get(key) is not None]
    hits = sorted(hits, key=lambda h: -h[key])
    return {**record, "results": [{**h, "rank": i} for i, h in enumerate(hits, start=1)]}


def evaluate_run(run_records, qrels_by_query, queries, run_name, exclude_query_types=()):
    rows_by_type = defaultdict(list)
    for rec in run_records:
        qid = rec["query_id"]
        if qid not in queries:
            continue
        qtype = queries[qid]["query_type"]
        if qtype in exclude_query_types:
            continue
        qrels = qrels_by_query.get(qid, {})
        ranked = ranked_ids_for(rec)
        row = {
            "precision_at_5": precision_at_k(ranked, qrels, 5),
            "precision_at_10": precision_at_k(ranked, qrels, 10),
            "recall_at_10": recall_at_k(ranked, qrels, 10),
            "recall_at_20": recall_at_k(ranked, qrels, 20),
            "map": average_precision(ranked, qrels),
            "ndcg_at_5": ndcg_at_k(ranked, qrels, 5),
            "ndcg_at_10": ndcg_at_k(ranked, qrels, 10),
            "judged_coverage": judged_coverage(ranked, qrels, 20),
        }
        rows_by_type[qtype].append(row)
        rows_by_type["all"].append(row)

    out = []
    for qtype, rows in rows_by_type.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        agg = df.mean().to_dict()
        agg["run_name"] = run_name
        agg["query_type"] = qtype
        agg["num_queries"] = len(rows)
        out.append(agg)
    return out


# ===========================================================================
# Qualitative error analysis
# ===========================================================================
def relevance_label(qrels, pid):
    if pid not in qrels:
        return "unjudged"
    return {0: "not relevant", 1: "partial", 2: "strong"}[qrels[pid]]


def format_hits(hits, qrels, k=5):
    lines = []
    for h in hits[:k]:
        pid = h["product_id"]
        score = h.get("final_score", h.get("prefusion_score"))
        lines.append(f"  {h['rank']}. {pid} (score={score:.3f}, relevance={relevance_label(qrels, pid)})")
    return "\n".join(lines) if lines else "  (no results)"


def relevant_rank(hits, qrels):
    """Rank of the first product with relevance > 0, or None."""
    for h in sorted(hits, key=lambda x: x["rank"]):
        if qrels.get(h["product_id"], 0) > 0:
            return h["rank"]
    return None


def build_error_analysis(retrieval, reranked, queries, qrels_by_query, path):
    retrieval_by_q = {r["query_id"]: r for r in retrieval}
    reranked_by_q = {r["query_id"]: r for r in reranked}

    cases = []

    # Case 1: dense retrieval beats sparse retrieval (dense finds a relevant
    # item earlier than sparse does).
    best_gap, best_qid = -1, None
    for qid, rec in retrieval_by_q.items():
        qrels = qrels_by_query.get(qid, {})
        dense_only = reorder_by(rec, "dense_score")
        sparse_only = reorder_by(rec, "sparse_score")
        dr, sr = relevant_rank(dense_only["results"], qrels), relevant_rank(sparse_only["results"], qrels)
        if dr is not None and (sr is None or sr > dr):
            gap = (sr or 999) - dr
            if gap > best_gap:
                best_gap, best_qid = gap, qid
    if best_qid:
        qid = best_qid
        q = queries[qid]
        qrels = qrels_by_query.get(qid, {})
        rec = retrieval_by_q[qid]
        cases.append((
            "Dense retrieval outperforms sparse retrieval",
            q, qrels,
            format_hits(reorder_by(rec, "dense_score")["results"], qrels),
            format_hits(reorder_by(rec, "sparse_score")["results"], qrels),
            "Dense embeddings capture visual/semantic similarity (style, shape, material look) "
            "that exact keyword overlap in SPLADE misses, so dense surfaces the relevant product "
            "earlier when the query uses paraphrased or visual language.",
        ))

    # Case 2: sparse beats dense.
    best_gap, best_qid = -1, None
    for qid, rec in retrieval_by_q.items():
        qrels = qrels_by_query.get(qid, {})
        dense_only = reorder_by(rec, "dense_score")
        sparse_only = reorder_by(rec, "sparse_score")
        dr, sr = relevant_rank(dense_only["results"], qrels), relevant_rank(sparse_only["results"], qrels)
        if sr is not None and (dr is None or dr > sr) and qid != best_qid:
            gap = (dr or 999) - sr
            if gap > best_gap:
                best_gap, best_qid = gap, qid
    if best_qid:
        qid = best_qid
        q = queries[qid]
        qrels = qrels_by_query.get(qid, {})
        rec = retrieval_by_q[qid]
        cases.append((
            "Sparse retrieval outperforms dense retrieval",
            q, qrels,
            format_hits(reorder_by(rec, "dense_score")["results"], qrels),
            format_hits(reorder_by(rec, "sparse_score")["results"], qrels),
            "The query names an exact attribute, brand, or product-type term. SPLADE's lexical "
            "term weighting matches that vocabulary directly, while dense similarity spreads "
            "mass across visually/semantically similar but attribute-mismatched products.",
        ))

    # Case 3: reranking improves ranking (relevant item moves up).
    best_gain, best_qid = -1, None
    for qid, rec in retrieval_by_q.items():
        rr = reranked_by_q.get(qid)
        if not rr:
            continue
        qrels = qrels_by_query.get(qid, {})
        before_rank = relevant_rank(rec["results"], qrels)
        after_rank = relevant_rank(rr["results"], qrels)
        if before_rank is not None and after_rank is not None and before_rank > after_rank:
            gain = before_rank - after_rank
            if gain > best_gain:
                best_gain, best_qid = gain, qid
    if best_qid:
        qid = best_qid
        q = queries[qid]
        qrels = qrels_by_query.get(qid, {})
        cases.append((
            "Cross-encoder reranking improves the ranking",
            q, qrels,
            format_hits(retrieval_by_q[qid]["results"], qrels),
            format_hits(reranked_by_q[qid]["results"], qrels),
            "The cross-encoder jointly attends to the query and each candidate's full text/image, "
            "so it can confirm fine-grained attribute matches (or negative-constraint compliance) "
            "that independent dense/sparse scoring could only approximate.",
        ))

    # Case 4: reranking hurts ranking (relevant item moves down).
    worst_loss, worst_qid = -1, None
    for qid, rec in retrieval_by_q.items():
        rr = reranked_by_q.get(qid)
        if not rr:
            continue
        qrels = qrels_by_query.get(qid, {})
        before_rank = relevant_rank(rec["results"], qrels)
        after_rank = relevant_rank(rr["results"], qrels)
        if before_rank is not None and after_rank is not None and after_rank > before_rank:
            loss = after_rank - before_rank
            if loss > worst_loss:
                worst_loss, worst_qid = loss, qid
    if worst_qid:
        qid = worst_qid
        q = queries[qid]
        qrels = qrels_by_query.get(qid, {})
        cases.append((
            "Cross-encoder reranking hurts the ranking",
            q, qrels,
            format_hits(retrieval_by_q[qid]["results"], qrels),
            format_hits(reranked_by_q[qid]["results"], qrels),
            "The reranker over-weighted surface text similarity (e.g. shared category words) "
            "over the visual/attribute mismatch that the dense and sparse scores had already "
            "correctly down-weighted, pulling a less relevant product to the top.",
        ))

    # Case 5: an image or image_text query, regardless of the metric behavior.
    img_qid = next((qid for qid, q in queries.items()
                     if q["query_type"] in ("image", "image_text") and qid in retrieval_by_q), None)
    if img_qid:
        qid = img_qid
        q = queries[qid]
        qrels = qrels_by_query.get(qid, {})
        rec = retrieval_by_q[qid]
        rr = reranked_by_q.get(qid, rec)
        cases.append((
            f"{q['query_type']} query behavior",
            q, qrels,
            format_hits(rec["results"], qrels),
            format_hits(rr["results"], qrels),
            "For image-bearing queries there is no independent text query to drive sparse "
            "retrieval, so the system relies on the single-vector multimodal dense embedding "
            "(and, for image_text queries, the text-modified combined query object) plus the "
            "multimodal cross-encoder to apply the requested visual/textual changes.",
        ))

    lines = ["# Stage 4 — Error Analysis\n"]
    for i, (title, q, qrels, before, after, explanation) in enumerate(cases, start=1):
        lines.append(f"## Case {i}: {title}\n")
        lines.append(f"**Query** `{q['query_id']}` ({q['query_type']}): "
                      f"{q.get('query_text') or q.get('query_image_path')}\n")
        lines.append("Top results before reranking:\n")
        lines.append(before + "\n")
        lines.append("Top results after reranking:\n")
        lines.append(after + "\n")
        lines.append(f"**What happened:** {explanation}\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return len(cases)


def main():
    ap = argparse.ArgumentParser(description="Stage 4: evaluation")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--qrels", default="data/qrels.jsonl")
    ap.add_argument("--runs", nargs="+",
                     default=["outputs/retrieval_results.jsonl", "outputs/reranked_results.jsonl"],
                     help="Retrieval run file(s) followed by the reranked run file")
    ap.add_argument("--out", default="reports/evaluation_metrics.csv")
    ap.add_argument("--report", default="reports/evaluation_report.md")
    ap.add_argument("--error_analysis", default="reports/error_analysis.md")
    args = ap.parse_args()

    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    qrels_by_query = load_qrels(args.qrels)

    retrieval_path = args.runs[0]
    reranked_path = args.runs[1] if len(args.runs) > 1 else None
    retrieval = load_jsonl(retrieval_path)

    all_rows = []
    dense_only_records = [reorder_by(r, "dense_score") for r in retrieval]
    sparse_only_records = [reorder_by(r, "sparse_score") for r in retrieval]

    all_rows += evaluate_run(dense_only_records, qrels_by_query, queries, "dense_single_vector")
    all_rows += evaluate_run(sparse_only_records, qrels_by_query, queries, "sparse_splade",
                              exclude_query_types=("image",))
    all_rows += evaluate_run(retrieval, qrels_by_query, queries, "hybrid_prefusion")

    reranked = []
    if reranked_path and os.path.exists(reranked_path):
        reranked = load_jsonl(reranked_path)
        ce_only_records = [reorder_by(r, "cross_encoder_score") for r in reranked]
        all_rows += evaluate_run(ce_only_records, qrels_by_query, queries, "hybrid_cross_encoder")
        all_rows += evaluate_run(reranked, qrels_by_query, queries, "hybrid_cross_encoder_fused")

    metrics_df = pd.DataFrame(all_rows)
    cols = ["run_name", "query_type", "num_queries", "precision_at_5", "precision_at_10",
            "recall_at_10", "recall_at_20", "map", "ndcg_at_5", "ndcg_at_10", "judged_coverage"]
    metrics_df = metrics_df[cols]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    metrics_df.to_csv(args.out, index=False)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("# Stage 4 — Evaluation Report\n\n")
        fh.write(f"Judged {len(qrels_by_query)} of {len(queries)} queries via a pooled qrels file "
                 f"(data/qrels.jsonl). Metrics: Precision@5/10, Recall@10/20, MAP, NDCG@5/10, "
                 f"and judged coverage (share of top-20 results that have a qrel label).\n\n")
        fh.write("## Comparison table\n\n")
        fh.write(metrics_df.round(4).to_markdown(index=False) + "\n\n")
        fh.write("## Limitations\n\n")
        fh.write("- Metrics rely on pooled qrels (top 10-20 unique products per query across "
                  "systems); unjudged products below the pool depth are treated as not relevant, "
                  "so recall is a lower bound.\n")
        fh.write("- Image-only queries have no sparse_splade or text-only cross-encoder ablation "
                  "rows, since there is no query text to drive sparse retrieval.\n")

    if reranked:
        n_cases = build_error_analysis(retrieval, reranked, queries, qrels_by_query, args.error_analysis)
    else:
        n_cases = 0

    print(metrics_df.round(4).to_string(index=False))
    print(f"\nWrote {args.out}, {args.report}, and {args.error_analysis} ({n_cases} cases)")


if __name__ == "__main__":
    main()
