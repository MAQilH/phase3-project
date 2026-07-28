import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def load_dense_model(model_name, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return model


def encode_products(model, catalog, batch_size):
    inputs = [
        {"text": row.product_text, "image": row.image_path}
        for row in catalog.itertuples()
    ]
    encode_fn = getattr(model, "encode_document", None) or model.encode
    used_encode_document = hasattr(model, "encode_document")
    vectors = encode_fn(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32), used_encode_document


def encode_query_dense(model, query_text, query_image_path):
    encode_fn = getattr(model, "encode_query", None) or model.encode
    if query_text and query_image_path:
        query_input = {"text": query_text, "image": query_image_path}
    elif query_image_path:
        query_input = query_image_path
    else:
        query_input = query_text
    vec = encode_fn([query_input], normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)[0]


class SpladeEncoder:
    def __init__(self, model_name, device, top_n=128):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.top_n = top_n
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.vocab_size = self.model.config.vocab_size

    def encode_batch(self, texts, max_length=256):
        torch = self.torch
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
            activations = torch.log1p(torch.relu(logits))
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled, _ = (activations * mask).max(dim=1)
        pooled = pooled.cpu().numpy()

        results = []
        for row in pooled:
            nz = np.nonzero(row)[0]
            if len(nz) > self.top_n:
                top_idx = nz[np.argpartition(-row[nz], self.top_n - 1)[: self.top_n]]
            else:
                top_idx = nz
            order = np.argsort(-row[top_idx])
            top_idx = top_idx[order]
            results.append((top_idx.tolist(), row[top_idx].astype(float).tolist()))
        return results

    def encode_products(self, texts, batch_size=16):
        out = []
        for i in range(0, len(texts), batch_size):
            out.extend(self.encode_batch(texts[i:i + batch_size]))
        return out

    def encode_query(self, text):
        return self.encode_batch([text])[0]


def build_dense_faiss(dense_matrix, path):
    import faiss
    dim = dense_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine == inner product on normalized vecs
    index.add(np.ascontiguousarray(dense_matrix, dtype=np.float32))
    faiss.write_index(index, path)
    return index


def build_sparse_csr(sparse_records, vocab_size, path):
    from scipy import sparse as sps
    rows, cols, vals = [], [], []
    for i, rec in enumerate(sparse_records):
        for idx, val in zip(rec["indices"], rec["values"]):
            rows.append(i)
            cols.append(idx)
            vals.append(val)
    mat = sps.csr_matrix((vals, (rows, cols)), shape=(len(sparse_records), vocab_size))
    sps.save_npz(path, mat)
    return mat


def dense_self_rank(dense_index, dense_matrix, probe_vector, probe_row, top_k=10):
    scores, idx = dense_index.search(np.ascontiguousarray([probe_vector], dtype=np.float32), top_k)
    idx = idx[0].tolist()
    return idx.index(probe_row) if probe_row in idx else None


def sparse_self_rank(sparse_matrix, query_indices, query_values, probe_row, vocab_size, top_k=10):
    q = np.zeros(vocab_size, dtype=np.float32)
    q[query_indices] = query_values
    scores = sparse_matrix.dot(q)
    order = np.argsort(-scores)[:top_k].tolist()
    return order.index(probe_row) if probe_row in order else None


def main():
    ap = argparse.ArgumentParser(description="Stage 1: dense + sparse embeddings and indexes")
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--dense_model", default="nvidia/llama-nemotron-embed-vl-1b-v2")
    ap.add_argument("--sparse_model", default="prithivida/Splade_PP_en_v1")
    ap.add_argument("--emb_dir", default="artifacts/embeddings")
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--report", default="reports/indexing_summary.md")
    ap.add_argument("--dense_batch_size", type=int, default=2)
    ap.add_argument("--sparse_batch_size", type=int, default=16)
    ap.add_argument("--sparse_top_n", type=int, default=128)
    ap.add_argument("--max_products", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")

    os.makedirs(args.emb_dir, exist_ok=True)
    os.makedirs(args.index_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    catalog = pd.read_parquet(args.catalog)
    debug_scale = args.max_products is not None and args.max_products < len(catalog)
    if debug_scale:
        catalog = catalog.iloc[: args.max_products].reset_index(drop=True)
    product_ids = catalog["product_id"].tolist()
    n = len(catalog)
    print(f"Embedding {n} products on device={device}"
          + (" (DEBUG SCALE RUN)" if debug_scale else ""))

    t0 = time.time()
    dense_model = load_dense_model(args.dense_model, device)
    dense_matrix, used_encode_document = encode_products(
        dense_model, catalog, args.dense_batch_size,
    )
    dense_time = time.time() - t0
    dense_dim = dense_matrix.shape[1]

    np.save(os.path.join(args.emb_dir, "product_dense.npy"), dense_matrix)
    with open(os.path.join(args.emb_dir, "product_ids.txt"), "w") as fh:
        fh.write("\n".join(product_ids) + "\n")

    t0 = time.time()
    sparse_encoder = SpladeEncoder(args.sparse_model, device, top_n=args.sparse_top_n)
    sparse_results = sparse_encoder.encode_products(
        catalog["product_text"].tolist(), batch_size=args.sparse_batch_size,
    )
    sparse_time = time.time() - t0

    sparse_records = []
    nnz_counts = []
    with open(os.path.join(args.emb_dir, "product_sparse.jsonl"), "w") as fh:
        for pid, (indices, values) in zip(product_ids, sparse_results):
            rec = {"product_id": pid, "indices": indices, "values": values}
            sparse_records.append(rec)
            nnz_counts.append(len(indices))
            fh.write(json.dumps(rec) + "\n")

    dense_index_path = os.path.join(args.index_dir, "dense_index.faiss")
    sparse_index_path = os.path.join(args.index_dir, "sparse_index.npz")
    dense_index = build_dense_faiss(dense_matrix, dense_index_path)
    sparse_matrix = build_sparse_csr(sparse_records, sparse_encoder.vocab_size, sparse_index_path)

    manifest = {
        "num_products": n,
        "dense_model": args.dense_model,
        "dense_input_format": "multimodal_dict_text_image",
        "dense_embedding_policy": "single_vector_from_combined_text_image_input",
        "manual_text_image_vector_mixing": False,
        "dense_encode_method": "encode_document" if used_encode_document else "encode",
        "sparse_model": args.sparse_model,
        "sparse_vocab_size": sparse_encoder.vocab_size,
        "dense_dimension": dense_dim,
        "sparse_top_n": args.sparse_top_n,
        "vector_store": "faiss_plus_csr",
        "device": device,
        "debug_scale_run": debug_scale,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.index_dir, "index_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    probe_rows = [0, n // 2, n - 1]
    sanity_lines = []
    for row in probe_rows:
        pid = product_ids[row]
        title = catalog.iloc[row]["title"]
        image_path = catalog.iloc[row]["image_path"]

        img_vec = encode_query_dense(dense_model, None, image_path)
        img_rank = dense_self_rank(dense_index, dense_matrix, img_vec, row)
        sanity_lines.append(f"DENSE image-query ({pid}): self rank={img_rank}")

        text_vec = encode_query_dense(dense_model, title, None)
        text_rank = dense_self_rank(dense_index, dense_matrix, text_vec, row)
        sanity_lines.append(f"DENSE text-query ({pid}): self rank={text_rank}")

        sp_idx, sp_val = sparse_encoder.encode_query(title)
        sp_rank = sparse_self_rank(sparse_matrix, sp_idx, sp_val, row, sparse_encoder.vocab_size)
        sanity_lines.append(f"SPARSE title-query ({pid}): self rank={sp_rank}")

    avg_nnz = float(np.mean(nnz_counts))
    report = f"""# Stage 1 — Indexing Summary

- Dense model: `{args.dense_model}` (loaded with `trust_remote_code=True`, device={device})
- Dense encode method: `{"encode_document" if used_encode_document else "encode"}`
- Dense input format: combined `{{"text": product_text, "image": image_path}}` dict, one forward
  pass per product -> one vector. No separate text/image encode + averaging is performed.
- Dense vector dimension: {dense_dim}
- Dense batch size: {args.dense_batch_size}; wall time: {dense_time:.1f}s for {n} products
- Sparse model: `{args.sparse_model}` (SPLADE++-style masked-LM, log(1+relu) + max pooling)
- Sparse top-n kept per product: {args.sparse_top_n}
- Average sparse non-zero count: {avg_nnz:.1f} (vocab size {sparse_encoder.vocab_size})
- Sparse batch size: {args.sparse_batch_size}; wall time: {sparse_time:.1f}s for {n} products
- Number of product vectors produced: {n}
- Index type: FAISS `IndexFlatIP` for dense (vectors L2-normalized -> inner product == cosine
  similarity); SciPy CSR sparse matrix for sparse (dot-product / cosine similarity).
- Runtime environment: {"DEBUG SCALE (" + str(n) + " products, local CPU/MPS smoke test)" if debug_scale else "full run"}

## Sanity-check searches

{chr(10).join(sanity_lines)}
"""
    with open(args.report, "w") as fh:
        fh.write(report)

    print(f"Wrote dense matrix {dense_matrix.shape}, sparse jsonl, indexes, manifest, and report.")
    print("\n".join(sanity_lines))


if __name__ == "__main__":
    main()
