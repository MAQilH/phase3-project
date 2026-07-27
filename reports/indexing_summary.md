# Stage 1 - Indexing Summary

## Dense embedding
- Model: `nvidia/llama-nemotron-embed-vl-1b-v2` (encode method=encode_document, device=cuda)
- Input format: combined {"text": product_text, "image": image_path} dict, one forward pass per product -> one vector (no separate encode + averaging)
- Dimension: 2048, batch size: 2, wall time: 0.0 min for 2000 products
- Vector norm mean/min/max: 1.0000 / 0.9996 / 1.0004

## Sparse embedding
- Model: `prithivida/Splade_PP_en_v1` (SPLADE++-style masked LM, log(1+relu) activations, max pooling)
- Vocab size: 30522, top-n kept per product: 128
- Stats: {'avg_nnz': 120.533, 'min_nnz': 46, 'max_nnz': 128, 'failures': 0}

## Indexes
- Dense: FAISS IndexFlatIP (2000 vectors, dim 2048); cosine similarity via inner product on L2-normalized vectors
- Sparse: SciPy CSR matrix (shape (2000, 30522)); dot-product similarity
- ID mapping file: artifacts/embeddings/product_ids.txt (row i <-> line i)

## Sanity-check searches
- DENSE image-query (product B07B51H7B1|amazon.com): self rank=0, top=['B07B51H7B1|amazon.com', 'B07FB9VR4P|amazon.com', 'B07B4W2MFW|amazon.com']
- DENSE text-query (product B07SYL8MPN|amazon.sa): self rank=0, top=['B07SYL8MPN|amazon.sa', 'B07SYL958P|amazon.sg', 'B07T25P4BB|amazon.com']
- DENSE image+text (product B075X25BYC|amazon.com): self rank=0, top=['B075X25BYC|amazon.com', 'B07QX2BYDH|amazon.com', 'B07QW3JRT4|amazon.com']
- SPARSE title-query (product B07B51H7B1|amazon.com): self rank=0, top=['B07B51H7B1|amazon.com', 'B075X2CVQK|amazon.com', 'B07CVC1Q7C|amazon.com']
- SPARSE title-query (product B07SYL8MPN|amazon.sa): self rank=0, top=['B07SYL8MPN|amazon.sa', 'B07SYL958P|amazon.sg', 'B07T25P4BB|amazon.com']
- SPARSE title-query (product B075X25BYC|amazon.com): self rank=0, top=['B075X25BYC|amazon.com', 'B07QX2BYDH|amazon.com', 'B07QW3JRT4|amazon.com']
