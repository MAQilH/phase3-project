"""Acceptance tests for Stage 0 (data prep) and Stage 1 (embeddings/indexes).

    # Stage 0 (the Stage 0 check) — run after prepare_data.py:
    python src/tests.py --stage 0 --catalog data/catalog_subset.parquet

    # Stage 1 (the Stage 1 checks) — run after the Colab notebook,
    # against the downloaded artifacts:
    python src/tests.py --stage 1 \
        --catalog data/catalog_subset.parquet \
        --emb_dir artifacts/embeddings \
        --index_dir artifacts/index \
        --summary reports/indexing_summary.md

faiss / scipy are optional for Stage 1: checks needing them are reported SKIP if
the package is missing (pip install faiss-cpu scipy to run them).

run_stage0(df) is also imported by prepare_data.py to gate the Stage 0 build.
"""

import argparse
import collections
import json
import os
import re

import numpy as np
import pandas as pd

# ---- Stage 1 manifest / summary expectations (the Stage 1 checks) ----
REQUIRED_MANIFEST_FIELDS = [
    "num_products", "dense_model", "dense_input_format",
    "dense_embedding_policy", "manual_text_image_vector_mixing",
    "sparse_model", "dense_dimension", "sparse_top_n", "vector_store",
    "created_at",
]
POLICY_EXPECTED = {
    "dense_input_format": "multimodal_dict_text_image",
    "dense_embedding_policy": "single_vector_from_combined_text_image_input",
    "manual_text_image_vector_mixing": False,
}
SUMMARY_KEYWORDS = [
    "dimension", "batch size", "runtime", "vectors", "top-n", "non-zero", "sanity",
]
SIMILARITY_TERMS = [
    "similarity", "cosine", "inner product", "dot product", "dot-product",
    "indexflatip", "index type",
]
ALLOWED_ROLES = {"exact", "substitute", "irrelevant"}
ALLOWED_DECISIONS = {
    "recommend_exact", "recommend_exact_with_warning",
    "recommend_substitute", "no_good_match", "ask_clarification",
}


class Checker:
    def __init__(self):
        self.ok = True
        self.n_pass = self.n_fail = self.n_skip = 0

    def check(self, name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        self.ok = self.ok and bool(cond)
        self.n_pass += bool(cond)
        self.n_fail += (not cond)

    def skip(self, name, why):
        print(f"  [SKIP] {name}  ({why})")
        self.n_skip += 1

    def summary(self):
        print(f"\n{self.n_pass} passed, {self.n_fail} failed, {self.n_skip} skipped")
        print("ALL PASSED" if self.ok else "SOME CHECKS FAILED")
        return self.ok


def load_optional(module):
    try:
        return __import__(module)
    except Exception:
        return None


# ===========================================================================
# Stage 0 (the Stage 0 check)
# ===========================================================================
def stage0_checks(df):
    return {
        "every row has a non-empty product_id":
            bool((df["product_id"].fillna("").str.len() > 0).all()),
        "product_id is unique": bool(df["product_id"].is_unique),
        "every row has non-empty title":
            bool((df["title"].fillna("").str.len() > 0).all()),
        "every row has non-empty product_text":
            bool((df["product_text"].fillna("").str.len() > 0).all()),
        "every row has non-empty image_path":
            bool((df["image_path"].fillna("").str.len() > 0).all()),
        "every local image_path exists":
            bool(df["image_path"].map(os.path.exists).all()),
        "subset contains multiple product types":
            bool(df["product_type"].nunique() > 1),
        "product_text contains metadata beyond the title":
            bool((df["product_text"].str.len() > df["title"].str.len()).mean() > 0.9),
    }


def run_stage0(df):
    print("Stage 0 acceptance tests:")
    c = Checker()
    for name, cond in stage0_checks(df).items():
        c.check(name, cond)
    return c.summary()


# ===========================================================================
# Stage 1 (the Stage 1 checks)
# ===========================================================================
def run_stage1(catalog_path, emb_dir, index_dir, summary_path):
    dense_npy = os.path.join(emb_dir, "product_dense.npy")
    ids_txt = os.path.join(emb_dir, "product_ids.txt")
    sparse_jsonl = os.path.join(emb_dir, "product_sparse.jsonl")
    dense_index = os.path.join(index_dir, "dense_index.faiss")
    sparse_index = os.path.join(index_dir, "sparse_index.npz")
    manifest_path = os.path.join(index_dir, "index_manifest.json")

    c = Checker()

    print("File existence:")
    for path in [catalog_path, dense_npy, ids_txt, sparse_jsonl, dense_index,
                 sparse_index, manifest_path, summary_path]:
        c.check(f"exists: {path}", os.path.exists(path))
    if not (os.path.exists(catalog_path) and os.path.exists(dense_npy)
            and os.path.exists(ids_txt)):
        print("\nCore artifacts missing — cannot continue.")
        return c.summary()

    catalog = pd.read_parquet(catalog_path)
    product_ids = catalog["product_id"].tolist()
    n = len(catalog)
    dense = np.load(dense_npy)
    ids_on_disk = open(ids_txt).read().split()

    print("\nDense embeddings:")
    c.check("exactly one dense vector per product", dense.shape[0] == n)
    c.check("product_ids.txt aligns row-for-row with catalog (row i <-> line i)",
            ids_on_disk == product_ids)
    c.check("product IDs in embeddings match catalog (as a set)",
            set(ids_on_disk) == set(product_ids))
    c.check("dense vectors L2-normalized (norm ~ 1.0)",
            bool(np.allclose(np.linalg.norm(dense, axis=1), 1.0, atol=1e-2)))

    print("\nIndex manifest:")
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        for field in REQUIRED_MANIFEST_FIELDS:
            c.check(f"manifest has '{field}'", field in manifest)
        for field, expected in POLICY_EXPECTED.items():
            c.check(f"manifest '{field}' == {expected!r}",
                    manifest.get(field) == expected)
        c.check("manifest num_products matches catalog",
                manifest.get("num_products") == n)
        c.check("manifest dense_dimension matches dense matrix",
                manifest.get("dense_dimension") == dense.shape[1])
    else:
        c.check("manifest exists", False)

    print("\nSparse embeddings:")
    sparse_rows, nnz_counts, failures = [], [], 0
    with open(sparse_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            sparse_rows.append(rec["product_id"])
            if rec.get("error") or not rec.get("indices"):
                failures += 1
            else:
                nnz_counts.append(len(rec["indices"]))
    c.check("one sparse record per product (vector or logged failure)",
            len(sparse_rows) == n)
    c.check("sparse product_ids align with catalog order",
            sparse_rows == product_ids)
    avg_nnz = float(np.mean(nnz_counts)) if nnz_counts else 0.0
    vocab = manifest.get("sparse_vocab_size")
    upper_ok = (avg_nnz < vocab) if vocab else (avg_nnz > 0)
    c.check(f"avg sparse nnz >0 and < vocab (avg={avg_nnz:.1f}, vocab={vocab})",
            0 < avg_nnz and upper_ok)
    top_n = manifest.get("sparse_top_n")
    if top_n:
        c.check(f"max sparse nnz <= sparse_top_n ({top_n})",
                (max(nnz_counts) if nnz_counts else 0) <= top_n)

    print("\nIndexing summary:")
    if os.path.exists(summary_path):
        text = open(summary_path).read().lower()
        dm = (manifest.get("dense_model") or "").lower()
        sm = (manifest.get("sparse_model") or "").lower()
        c.check("summary names the dense model", bool(dm) and dm in text)
        c.check("summary names the sparse model", bool(sm) and sm in text)
        for kw in SUMMARY_KEYWORDS:
            c.check(f"summary mentions '{kw}'", kw in text)
        c.check("summary states the similarity / index function",
                any(t in text for t in SIMILARITY_TERMS))
        sanity_lines = re.findall(r"self rank=([0-9]+|none)", text, flags=re.IGNORECASE)
        c.check("summary has >= 3 sanity-check searches", len(sanity_lines) >= 3)
    else:
        c.check("summary exists", False)

    print("\nRetrieval self-checks (from summary):")
    summary_text = open(summary_path).read() if os.path.exists(summary_path) else ""

    def self_rank_for(kind):
        m = re.search(re.escape(kind) + r".*?self rank=([0-9]+|None)",
                      summary_text, flags=re.IGNORECASE)
        if not m:
            return None, False
        val = m.group(1)
        return (None if val.lower() == "none" else int(val)), True

    for kind in ["DENSE image-query", "DENSE text-query", "SPARSE title-query"]:
        rank, found = self_rank_for(kind)
        if not found:
            c.skip(f"{kind} self-retrieval in top-5", "not found in summary")
        else:
            c.check(f"{kind} self-retrieval in top-5 (rank={rank})",
                    rank is not None and rank < 5)

    print("\nDense index consistency:")
    faiss = load_optional("faiss")
    if faiss is None:
        c.skip("dense index self-probe", "faiss not installed")
    elif not os.path.exists(dense_index):
        c.check("dense index file present for probe", False)
    else:
        idx = faiss.read_index(dense_index)
        c.check("FAISS ntotal == num products", idx.ntotal == n)
        probes = [0, n // 2, n - 1]
        q = np.ascontiguousarray(dense[probes], dtype=np.float32)
        _, nn = idx.search(q, 1)
        c.check("stored vector retrieves itself at rank 0",
                all(int(nn[i][0]) == probes[i] for i in range(len(probes))))

    print("\nSparse index shape:")
    if load_optional("scipy") is None:
        c.skip("sparse_index.npz shape", "scipy not installed")
    elif not os.path.exists(sparse_index):
        c.check("sparse index file present", False)
    else:
        from scipy import sparse as sps
        mat = sps.load_npz(sparse_index)
        c.check("sparse matrix rows == num products", mat.shape[0] == n)
        if vocab:
            c.check("sparse matrix cols == vocab size", mat.shape[1] == vocab)

    return c.summary()


# ===========================================================================
# Stage 2 / Stage 3 (the Stage 2 and Stage 3 schema checks) — schema-level checks.
# These do NOT load the heavy models, so they are safe to run anywhere the
# output JSONL files exist (i.e. on the same machine that ran retrieve.py
# and rerank.py, or on a laptop that downloaded only the JSONL outputs).
# ===========================================================================
def _check_jsonl(path, c):
    if not os.path.exists(path):
        c.check(f"file exists: {path}", False)
        return []
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(ln) for ln in fh if ln.strip()]
    c.check(f"file non-empty: {path} ({len(records)} records)", bool(records))
    return records


def _validate_hits(results, expect_ce, c):
    """Per-record hit checks."""
    if not results:
        c.check("at least one result", False)
        return
    ranks = [h["rank"] for h in results]
    c.check("ranks are consecutive 1..N",
            ranks == list(range(1, len(ranks) + 1)))
    ids = [h["product_id"] for h in results]
    c.check("no duplicate product_id in results", len(set(ids)) == len(ids))
    for need in ("rank", "product_id", "source"):
        c.check(f"every hit has '{need}'", all(need in h for h in results))
    c.check("dense_score is numeric or null",
            all(h.get("dense_score") is None
                or isinstance(h.get("dense_score"), (int, float))
                for h in results))
    # prefusion_score must be present in Stage 2
    c.check("every hit has prefusion_score",
            all("prefusion_score" in h for h in results))
    if expect_ce:
        c.check("every hit has cross_encoder_score (Stage 3)",
                all("cross_encoder_score" in h for h in results))
        c.check("every hit has final_score (Stage 3)",
                all("final_score" in h for h in results))


def run_stage2(retrieval_path, queries_path, catalog_path):
    c = Checker()
    print("Stage 2 acceptance tests:")
    records = _check_jsonl(retrieval_path, c)
    if not records:
        return c.summary()

    catalog = pd.read_parquet(catalog_path)
    catalog_ids = set(catalog["product_id"].tolist())
    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            q = json.loads(ln)
            queries[q["query_id"]] = q

    for r in records:
        c.check(f"{r['query_id']} exists in queries.jsonl",
                r["query_id"] in queries)
        q = queries.get(r["query_id"], {})
        c.check(f"{r['query_id']} query_type matches",
                r.get("query_type") == q.get("query_type"))
        c.check(f"{r['query_id']} run_name is hybrid prefusion",
                r.get("run_name") == "hybrid_dense_sparse_prefusion")
        c.check(f"{r['query_id']} sparse_applicable matches query_type",
                r.get("sparse_applicable") == (q.get("query_type") != "image"))
        # sparse_score must be null for image-only
        if q.get("query_type") == "image":
            c.check(f"{r['query_id']} image-only has no sparse_score",
                    all(h.get("sparse_score") is None
                        for h in r.get("results", [])))
        else:
            c.check(f"{r['query_id']} has sparse_score for text-bearing",
                    any(h.get("sparse_score") is not None
                        for h in r.get("results", [])))
        for h in r.get("results", []):
            c.check(f"{r['query_id']} product {h['product_id']} exists in catalog",
                    h["product_id"] in catalog_ids)
        _validate_hits(r.get("results", []), expect_ce=False, c=c)

    return c.summary()


def run_stage3(reranked_path, retrieval_path, queries_path, catalog_path):
    c = Checker()
    print("Stage 3 acceptance tests:")
    records = _check_jsonl(reranked_path, c)
    if not records:
        return c.summary()
    retrieval = {r["query_id"]: r for r in _check_jsonl(retrieval_path, c)}
    catalog = pd.read_parquet(catalog_path)
    catalog_ids = set(catalog["product_id"].tolist())
    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            q = json.loads(ln)
            queries[q["query_id"]] = q

    for r in records:
        c.check(f"{r['query_id']} exists in retrieval",
                r["query_id"] in retrieval)
        if r["query_id"] in retrieval:
            ret_ids = {h["product_id"] for h in retrieval[r["query_id"]]["results"]}
            rer_ids = {h["product_id"] for h in r["results"]}
            c.check(f"{r['query_id']} rerank subset of retrieval candidate pool",
                    rer_ids.issubset(ret_ids))
        for h in r.get("results", []):
            c.check(f"{r['query_id']} product {h['product_id']} in catalog",
                    h["product_id"] in catalog_ids)
        _validate_hits(r.get("results", []), expect_ce=True, c=c)
    return c.summary()


# ===========================================================================
# Stage 4 / Stage 5 (the Stage 4 and Stage 5 checks) — output-level acceptance checks.
# These checks validate the generated reports and structured answer schema.
# They do not call LLMs or load retrieval models.
# ===========================================================================
def _load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            q = json.loads(ln)
            queries[q["query_id"]] = q
    return queries


def run_stage4(catalog_path, queries_path, qrels_path, metrics_path,
               eval_report_path, error_analysis_path):
    c = Checker()
    print("Stage 4 acceptance tests:")
    from evaluate import (
        average_precision, ndcg_at_k, precision_at_k, recall_at_k,
    )

    toy_ranked = ["a", "b", "c"]
    toy_qrels = {"a": 2, "c": 1, "x": 1}
    c.check("toy Precision@2 matches hand calculation",
            abs(precision_at_k(toy_ranked, toy_qrels, 2) - 0.5) < 1e-9)
    c.check("toy Recall@3 matches hand calculation",
            abs(recall_at_k(toy_ranked, toy_qrels, 3) - (2 / 3)) < 1e-9)
    c.check("toy AP matches hand calculation",
            abs(average_precision(toy_ranked, toy_qrels) - (5 / 9)) < 1e-9)
    c.check("toy NDCG@3 matches hand calculation",
            abs(ndcg_at_k(toy_ranked, toy_qrels, 3) - 0.8472668888) < 1e-8)
    c.check("missing qrels are handled gracefully",
            recall_at_k(toy_ranked, {}, 3) == 0.0
            and ndcg_at_k(toy_ranked, {}, 3) == 0.0)
    catalog = pd.read_parquet(catalog_path)
    catalog_ids = set(catalog["product_id"].tolist())
    queries = _load_queries(queries_path)
    query_ids = set(queries)

    print("\nqrels:")
    qrels = _check_jsonl(qrels_path, c)
    required = {"query_id", "product_id", "relevance", "reason"}
    c.check("every qrel has required fields",
            all(required.issubset(r) for r in qrels))
    pairs = [(r.get("query_id"), r.get("product_id")) for r in qrels]
    c.check("no duplicate (query_id, product_id) qrels",
            len(pairs) == len(set(pairs)))
    c.check("qrel relevance is one of 0, 1, 2",
            all(r.get("relevance") in {0, 1, 2} for r in qrels))
    c.check("every qrel query_id exists in queries",
            all(r.get("query_id") in query_ids for r in qrels))
    c.check("every qrel product_id exists in catalog",
            all(r.get("product_id") in catalog_ids for r in qrels))
    c.check("qrels cover at least one text query",
            any(queries.get(r.get("query_id"), {}).get("query_type") == "text"
                for r in qrels))
    qrel_counts = collections.Counter(r.get("query_id") for r in qrels)
    c.check("qrels cover every query",
            set(qrel_counts) == query_ids)
    c.check("each query has a reasonably sized judgment pool (10-50)",
            all(10 <= qrel_counts.get(query_id, 0) <= 50
                for query_id in query_ids))
    c.check("qrels are human judgments, not auto-generated labels",
            all(not str(r.get("reason", "")).lower().startswith("auto:")
                for r in qrels))

    print("\nmetrics csv:")
    c.check(f"exists: {metrics_path}", os.path.exists(metrics_path))
    metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    need_cols = {
        "run_name", "query_type", "num_queries", "precision_at_5",
        "precision_at_10", "recall_at_10", "recall_at_20", "map",
        "ndcg_at_5", "ndcg_at_10", "judged_coverage",
    }
    c.check("metrics csv has required columns",
            need_cols.issubset(set(metrics.columns)))
    if not metrics.empty:
        required_runs = {
            "dense_single_vector", "sparse_splade", "hybrid_prefusion",
            "hybrid_cross_encoder", "hybrid_cross_encoder_fused",
        }
        c.check("metrics include all five required comparison runs",
                required_runs.issubset(set(metrics["run_name"])))
        c.check("metrics include text/image/image_text/all rows",
                {"text", "image", "image_text", "all"}.issubset(
                    set(metrics["query_type"])))
        sparse_rows = metrics[metrics["run_name"] == "sparse_splade"]
        c.check("sparse-only metrics exclude image-only queries",
                "image" not in set(sparse_rows["query_type"]))
        ce_rows = metrics[metrics["run_name"] == "hybrid_cross_encoder"]
        c.check("text-only cross-encoder metrics exclude image-only queries",
                "image" not in set(ce_rows["query_type"]))
        metric_cols = [
            "precision_at_5", "precision_at_10", "recall_at_10",
            "recall_at_20", "map", "ndcg_at_5", "ndcg_at_10",
            "judged_coverage",
        ]
        c.check("metric values are in [0, 1]",
                bool(((metrics[metric_cols] >= 0)
                      & (metrics[metric_cols] <= 1)).all().all()))
        c.check("num_queries are positive",
                bool((metrics["num_queries"] > 0).all()))

    print("\nreports:")
    c.check(f"exists: {eval_report_path}", os.path.exists(eval_report_path))
    if os.path.exists(eval_report_path):
        text = open(eval_report_path, encoding="utf-8").read().lower()
        for kw in ["stage 4", "map", "ndcg", "judged coverage", "qrels",
                   "limitations"]:
            c.check(f"evaluation report mentions '{kw}'", kw in text)
    c.check(f"exists: {error_analysis_path}", os.path.exists(error_analysis_path))
    if os.path.exists(error_analysis_path):
        text = open(error_analysis_path, encoding="utf-8").read()
        headings = re.findall(r"^##\s+", text, flags=re.MULTILINE)
        lower = text.lower()
        c.check("error analysis has at least five qualitative cases",
                len(headings) >= 5)
        c.check("error analysis shows top results before reranking",
                "top results before reranking" in lower)
        c.check("error analysis shows top results after reranking",
                "top results after reranking" in lower)
        c.check("error analysis includes image or image+text case",
                "image" in lower)
    return c.summary()


def run_stage5(answers_path, reranked_path, queries_path, catalog_path,
               summary_path, top_k=8):
    c = Checker()
    print("Stage 5 acceptance tests:")
    catalog = pd.read_parquet(catalog_path)
    catalog_ids = set(catalog["product_id"].tolist())
    queries = _load_queries(queries_path)
    reranked = {r["query_id"]: r for r in _check_jsonl(reranked_path, c)}
    answers = _check_jsonl(answers_path, c)

    c.check("one final answer per query",
            len(answers) == len(queries)
            and {a.get("query_id") for a in answers} == set(queries))
    required = {
        "query_id", "interpreted_need", "product_judgements",
        "decision", "customer_response",
    }
    forbidden_claims = ["price", "stock", "review", "rating", "shipping",
                        "discount"]
    for a in answers:
        qid = a.get("query_id")
        c.check(f"{qid} answer has required fields",
                required.issubset(a))
        c.check(f"{qid} query exists", qid in queries)
        c.check(f"{qid} decision label allowed",
                a.get("decision") in ALLOWED_DECISIONS)
        c.check(f"{qid} interpreted_need is an object",
                isinstance(a.get("interpreted_need"), dict))
        need = a.get("interpreted_need") or {}
        c.check(f"{qid} interpreted_need category is a string",
                isinstance(need.get("category"), str))
        for field in ["positive_preferences", "negative_constraints",
                      "visual_preferences", "uncertain_fields"]:
            c.check(f"{qid} interpreted_need {field} is a list",
                    isinstance(need.get(field), list))
        judgements = a.get("product_judgements")
        c.check(f"{qid} product_judgements is a non-empty list",
                isinstance(judgements, list) and bool(judgements))
        hit_ids = {
            h["product_id"]
            for h in reranked.get(qid, {}).get("results", [])[:top_k]
        }
        c.check(f"{qid} has reranked top-{top_k} candidates",
                bool(hit_ids))
        if isinstance(judgements, list):
            c.check(f"{qid} judgement product_ids are in top-{top_k}",
                    all(j.get("product_id") in hit_ids for j in judgements))
            c.check(f"{qid} judgement product_ids exist in catalog",
                    all(j.get("product_id") in catalog_ids for j in judgements))
            c.check(f"{qid} judgement roles allowed",
                    all(j.get("role") in ALLOWED_ROLES for j in judgements))
            c.check(f"{qid} judgement evidence is a list",
                    all(isinstance(j.get("evidence"), list)
                        for j in judgements))
            c.check(f"{qid} constraint violations are lists",
                    all(isinstance(j.get("constraint_violations"), list)
                        for j in judgements))
            c.check(f"{qid} judgements have reasons",
                    all(isinstance(j.get("reason"), str) and j.get("reason")
                        for j in judgements))
        response = a.get("customer_response")
        c.check(f"{qid} customer_response is understandable",
                isinstance(response, str) and len(response.split()) >= 8)
        response_l = (response or "").lower()
        c.check(f"{qid} response avoids unavailable commerce facts",
                not any(term in response_l for term in forbidden_claims))

    print("\nsummary:")
    c.check(f"exists: {summary_path}", os.path.exists(summary_path))
    if os.path.exists(summary_path):
        text = open(summary_path, encoding="utf-8").read().lower()
        for kw in ["stage 5", "llm", "json validity", "schema-valid",
                   "fallback", "example final answers", "limitations"]:
            c.check(f"llm summary mentions '{kw}'", kw in text)
    return c.summary()


def main():
    ap = argparse.ArgumentParser(
        description="Stage 0 / 1 / 2 / 3 / 4 / 5 acceptance tests")
    ap.add_argument("--stage", required=True,
                    choices=["0", "1", "2", "3", "4", "5"])
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--emb_dir", default="artifacts/embeddings")
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--summary", default="reports/indexing_summary.md")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--qrels", default="data/qrels.jsonl")
    ap.add_argument("--metrics", default="reports/evaluation_metrics.csv")
    ap.add_argument("--eval_report", default="reports/evaluation_report.md")
    ap.add_argument("--error_analysis", default="reports/error_analysis.md")
    ap.add_argument("--answers", default="outputs/final_answers.jsonl")
    ap.add_argument("--llm_summary", default="reports/llm_output_summary.md")
    ap.add_argument("--top_k", type=int, default=8)
    args = ap.parse_args()

    if args.stage == "0":
        df = pd.read_parquet(args.catalog)
        ok = run_stage0(df)
        print(f"Loadable by pandas.read_parquet: True ({len(df)} rows)")
    elif args.stage == "1":
        ok = run_stage1(args.catalog, args.emb_dir, args.index_dir, args.summary)
    elif args.stage == "2":
        ok = run_stage2(args.retrieval, args.queries, args.catalog)
    elif args.stage == "3":
        ok = run_stage3(args.reranked, args.retrieval, args.queries, args.catalog)
    elif args.stage == "4":
        ok = run_stage4(args.catalog, args.queries, args.qrels,
                        args.metrics, args.eval_report, args.error_analysis)
    else:
        ok = run_stage5(args.answers, args.reranked, args.queries,
                        args.catalog, args.llm_summary, top_k=args.top_k)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
