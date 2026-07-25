# Stage 1 — Indexing Summary

- Dense model: `nvidia/llama-nemotron-embed-vl-1b-v2` (loaded with `trust_remote_code=True`, device=mps)
- Dense encode method: `encode_document`
- Dense input format: combined `{"text": product_text, "image": image_path}` dict, one forward
  pass per product -> one vector. No separate text/image encode + averaging is performed.
- Dense vector dimension: 2048
- Dense batch size: 2; wall time: 125.2s for 60 products
- Sparse model: `prithivida/Splade_PP_en_v1` (SPLADE++-style masked-LM, log(1+relu) + max pooling)
- Sparse top-n kept per product: 128
- Average sparse non-zero count: 121.6 (vocab size 30522)
- Sparse batch size: 16; wall time: 5.9s for 60 products
- Number of product vectors produced: 60
- Index type: FAISS `IndexFlatIP` for dense (vectors L2-normalized -> inner product == cosine
  similarity); SciPy CSR sparse matrix for sparse (dot-product / cosine similarity).
- Runtime environment: DEBUG SCALE (60 products, local CPU/MPS smoke test)

## Sanity-check searches

DENSE image-query (B07B51H7B1|amazon.com): self rank=0
DENSE text-query (B07B51H7B1|amazon.com): self rank=0
SPARSE title-query (B07B51H7B1|amazon.com): self rank=0
DENSE image-query (B001ELJB5O|amazon.com): self rank=0
DENSE text-query (B001ELJB5O|amazon.com): self rank=0
SPARSE title-query (B001ELJB5O|amazon.com): self rank=0
DENSE image-query (B07SVJB925|amazon.co.uk): self rank=0
DENSE text-query (B07SVJB925|amazon.co.uk): self rank=0
SPARSE title-query (B07SVJB925|amazon.co.uk): self rank=0
