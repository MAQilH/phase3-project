# Phase 3 Information Retrieval Project: Hybrid Multimodal Product Search

**Course:** Information Retrieval  
**Project theme:** Product search with single-vector multimodal dense retrieval, SPLADE++-style sparse retrieval, cross-encoder fusion, evaluation, and structured language-model output  
**Recommended environment:** Google Colab or Kaggle Notebook with GPU  
**Dataset:** A category-restricted subset of Amazon Berkeley Objects (ABO)  
**Last updated:** 2026-06-19

---

## 1. Project Summary

In this project, students build a realistic product search pipeline. A user may search for a product using a text query, an image query, or a query that contains both text and an image. The system must retrieve candidate products from a product catalog, improve the ranking using a cross-encoder reranker, evaluate retrieval quality, and finally generate a structured JSON answer using a small language model.

The required search system is **hybrid and multimodal**:

1. Each product must have exactly one **single-vector multimodal dense embedding** produced from both product text and product image together.
2. The dense model must accept an image, text, or a combined image+text object and return one embedding vector for that input. Do not build the final product vector by separately encoding text and image and manually averaging them.
3. Each product must also have a **sparse neural embedding** based on product text, preferably using a SPLADE++-style model.
4. Candidate generation must use both dense and sparse retrieval whenever the query contains text.
5. Dense and sparse candidate evidence must be fused and reranked using a cross-encoder or multimodal reranker.
6. The system must be evaluated before and after reranking.
7. The final output must be valid JSON that explains the interpreted user need, judges retrieved products, and gives a user-facing response.

The project is divided into stages with strict input and output contracts. These contracts make grading easier: each stage produces files with stable schemas, and later stages consume those files without relying on hidden notebook state.

## 1.1 Release Package Setup

The student package is intentionally small. After unzipping it, work from the `project/` directory:

```text
project/
  phase3_hybrid_multimodal_product_search_assignment.md
  air_phase3_complete_todo.ipynb
  pyproject.toml
  src/
  data/
    catalog_subset.parquet
    queries.jsonl
    prep_stats.json
    AIR_Phase3_stage1_images.zip
```

Extract the image archive once before running the catalog check:

```bash
unzip -q data/AIR_Phase3_stage1_images.zip -d .
uv sync
uv run python src/tests.py --stage 0 --catalog data/catalog_subset.parquet
```

Use `uv sync --extra stage1-gpu` for the embedding, retrieval, and reranking runtime. Use `uv sync --extra stage5-llm` in a separate GPU runtime for the local language-model section. The notebook has setup cells for Colab/Kaggle as well; they read dependency groups from `pyproject.toml`.

---

## 2. Learning Goals

By the end of the project, students should be able to:

- Clean and normalize real product metadata.
- Build a Colab/Kaggle-friendly subset of a large product dataset.
- Generate single-vector multimodal dense embeddings from text+image product inputs.
- Generate sparse neural retrieval vectors using SPLADE++-style models.
- Store and query dense and sparse vectors using a vector database or local index.
- Implement text search, image search, and image+text search.
- Fuse dense, sparse, and cross-encoder evidence.
- Evaluate retrieval systems using Precision@K, Recall@K, MAP, and NDCG.
- Produce reliable structured JSON output with a small language model.
- Analyze system errors involving negation, visual similarity, exact attributes, and category mismatch.

---

## 3. Required Dataset: ABO-Home-2K

### 3.1 Source Dataset

Use the **Amazon Berkeley Objects (ABO)** dataset:

```text
https://amazon-berkeley-objects.s3.amazonaws.com/index.html
```

For this project, use only:

- product listing metadata;
- image metadata;
- small catalog images, where the largest image axis is at most 256 pixels.

Do **not** use the full-resolution image archive, 360-degree image archive, or 3D model archive for the base project. They are too large for standard Colab/Kaggle runs.

Useful dataset locations:

```text
Dataset home:
https://amazon-berkeley-objects.s3.amazonaws.com/index.html

Listings archive:
https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar

Small images archive:
https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar

Image metadata S3 directory:
s3://amazon-berkeley-objects/images/metadata/

AWS S3 bucket:
s3://amazon-berkeley-objects/
```

ABO provides product metadata, catalog images, and 3D assets under CC BY 4.0. The dataset page lists the product listings archive as about 83 MB and the downscaled small-image archive as about 3 GB. These sizes make a category-restricted subset realistic in notebook environments.

### 3.2 Required Course Subset

The full dataset is larger than needed for this assignment. Build a deterministic restricted subset called:

```text
ABO-Home-2K
```

The release package already includes the frozen ABO-Home-2K catalog, query file, and selected small images needed for the base run. Use those files for your submitted pipeline. The filtering notes below document how the subset was constructed and are useful if you want to reproduce or extend it.

Use a home, furniture, and household-goods subset because it contains visually meaningful products and metadata attributes such as product type, color, material, style, and dimensions.

Use this filtering strategy unless the teaching staff provides a frozen product-ID list:

1. Load all listing metadata files: `listings/metadata/listings_*.json.gz`.
2. Keep only products with:
   - a non-empty English title if available;
   - a non-empty `main_image_id`;
   - at least one usable category path or product type;
   - a category path containing at least one of:
     - `Furniture`
     - `Home & Kitchen`
     - `Home Decor`
     - `Home Décor`
     - `Lighting`
     - `Bedding`
     - `Storage & Organization`
     - `Kitchen & Dining`
3. Join with image metadata using `main_image_id = image_id`.
4. Keep only rows where the small image is available locally or can be downloaded from the small-image S3 path.
5. Stratify by `product_type` and sample at most 80 products per product type.
6. Use random seed `42`.
7. Stop at 2,000 products by default.

Recommended size settings:

| Setting | Products | Use case |
|---|---:|---|
| Debug | 200 | Fast schema and pipeline testing |
| Base | 1,000-2,000 | Required project scale |
| Stretch | 5,000 | Stronger experiment if runtime allows |

### 3.3 Normalized Catalog Schema

Stage 0 must produce:

```text
data/catalog_subset.parquet
```

Required columns:

| Column | Type | Description |
|---|---:|---|
| `product_id` | string | Unique product key. Use `item_id + "|" + domain_name`, not only `item_id`. |
| `item_id` | string | ABO item ID. |
| `domain_name` | string | Marketplace domain. |
| `title` | string | Main English product title if available; otherwise best available title. |
| `brand` | string or null | Brand name if available. |
| `product_type` | string or null | ABO product type. |
| `category_path` | string or null | Best category path from `node`. |
| `color` | string or null | Normalized color if available. |
| `material` | string or null | Material if available. |
| `style` | string or null | Style if available. |
| `dimensions` | string or null | Short normalized dimensions string if available. |
| `description` | string or null | Cleaned product description, HTML removed. |
| `bullet_points` | string | Concatenated English bullet points. |
| `image_id` | string | Main product image ID. |
| `image_path` | string | Local path to the 256px image. |
| `product_text` | string | Searchable text built from title, category, brand, attributes, bullets, and description. |

Recommended additional columns:

| Column | Type | Description |
|---|---:|---|
| `has_image` | bool | Whether local image file is available. |
| `text_length` | int | Number of characters or tokens in `product_text`. |
| `source_split` | string | `train`, `dev`, or `test` if fixed splits are created. |
| `metadata_json` | string | Optional compact JSON string with extra fields. |

### 3.4 Product Text Construction

Build `product_text` using a stable template:

```text
Title: {title}
Category: {category_path}
Product type: {product_type}
Brand: {brand}
Color: {color}
Material: {material}
Style: {style}
Dimensions: {dimensions}
Description: {description}
Features: {bullet_points}
```

Rules:

- Remove HTML tags from descriptions.
- Prefer English fields with language tags such as `en_US`, `en_GB`, or `en_IN`.
- Keep useful words such as color, material, product type, style, and dimensions.
- Do not add information that is not present in the metadata.
- Truncate very long text to a reasonable length, such as 256-512 tokens for embedding and reranking models.

---

## 4. End-to-End Architecture

```text
Raw ABO metadata + small images
        |
        v
[Stage 0] Data preparation and subset creation
        |
        v
data/catalog_subset.parquet
        |
        v
[Stage 1] Single-vector multimodal dense embeddings + SPLADE++ sparse embeddings + indexes
        |
        v
Dense index + sparse index + embedding artifacts
        |
        v
[Stage 2] Text / image / image+text candidate retrieval
        |
        v
outputs/retrieval_results.jsonl
        |
        v
[Stage 3] Cross-encoder fusion and reranking
        |
        v
outputs/reranked_results.jsonl
        +--------------------+
        |                    |
        v                    v
[Stage 4] Evaluation   [Stage 5] Structured LLM response
        |                    |
        v                    v
reports/evaluation.*    outputs/final_answers.jsonl
```

Every stage must save its outputs to disk. A notebook-only implementation is not acceptable unless it writes all required artifacts and can be rerun from top to bottom.

---

# Stage 0 — Data Preparation

## 5. Goal

Create a clean, small, reproducible product catalog from ABO that can be used by the rest of the project.

## 5.1 Required Input

One of the following raw input configurations:

```text
Raw ABO archives:
- abo-listings.tar
- abo-images-small.tar
- images metadata files from images/metadata/
```

or:

```text
Teaching-staff-provided raw sample:
- data/raw_abo_sample.parquet
```

## 5.2 Required Output

```text
data/catalog_subset.parquet
reports/data_preparation_summary.md
```

## 5.3 Minimum Functional Requirements

The preprocessing code must:

- parse ABO listing metadata;
- select the required home/furniture/household subset;
- join listing metadata with image metadata;
- copy or download only the images needed for the subset;
- create the required unique `product_id`;
- create a clean `product_text` field;
- log missing fields and skipped products;
- save the final catalog as Parquet.

CSV may also be saved for debugging, but Parquet is the required official artifact.

## 5.4 Suggested CLI

```bash
python src/prepare_data.py \
  --listings_dir data/raw/listings/metadata \
  --images_metadata_dir data/raw/images/metadata \
  --small_images_dir data/raw/images/small \
  --category_mode home \
  --max_products 2000 \
  --seed 42 \
  --out data/catalog_subset.parquet
```

## 5.5 Data Quality Report

`reports/data_preparation_summary.md` must include:

- number of raw products scanned;
- number of products after each filter;
- number of final products;
- number of unique product types;
- missing title rate;
- missing image rate;
- examples of 10 final products;
- distribution of `product_type` values;
- average and median `product_text` length;
- a short statement that the large original-image and 3D archives were not used.

## 5.6 Acceptance Tests

Stage 0 passes basic checks if:

- every row has a non-empty `product_id`;
- `product_id` is unique;
- every row has non-empty `title`, `product_text`, and `image_path`;
- every local `image_path` exists;
- the subset contains multiple product types;
- `product_text` contains useful metadata beyond the title;
- the output can be loaded by `pandas.read_parquet`.

## 5.7 Common Mistakes

- Using only `item_id` as product ID. Use `item_id|domain_name`.
- Forgetting to remove HTML from descriptions.
- Keeping products with missing local images.
- Accidentally downloading the 110 GB original image archive.
- Creating a subset with only one product type, which makes retrieval evaluation uninformative.

---

# Stage 1 — Single-Vector Multimodal Dense Embeddings, Sparse Embeddings, and Indexes

## 6. Goal

Build the first-stage search representation. Each product must have:

1. one **single-vector multimodal dense embedding** generated from the combined product text and product image;
2. one **sparse neural embedding** generated from product text using a SPLADE++-style model;
3. searchable dense and sparse indexes.

The dense product vector must be produced directly from a multimodal input such as:

```python
{"text": product_text, "image": image_path}
```

The final dense vector must **not** be created by separately encoding text and image and averaging, concatenating, or manually weighting those two vectors.

## 6.1 Required Input

```text
data/catalog_subset.parquet
```

Minimum required columns:

```text
product_id, title, product_text, image_path
```

## 6.2 Required Output

```text
artifacts/embeddings/product_dense.npy
artifacts/embeddings/product_sparse.jsonl      # or .npz / CSR / Qdrant sparse vectors
artifacts/embeddings/product_ids.txt
artifacts/index/dense_index.*
artifacts/index/sparse_index.*
artifacts/index/index_manifest.json
reports/indexing_summary.md
```

If using Qdrant, the index artifacts may be a local Qdrant storage directory plus an index manifest.

## 6.3 Required Dense Embedding Method

Use a multimodal embedding model that can accept text, image, or a combined image+text object and return one embedding vector. Recommended model families include the multimodal Sentence Transformers models described in the Hugging Face multimodal Sentence Transformers guide.

Recommended dense model choices:

| Model | Suggested role | Notes |
|---|---|---|
| `nvidia/llama-nemotron-embed-vl-1b-v2` | Recommended base | Smaller than many VLM embedders and suitable for Kaggle/Colab GPUs with careful batching. |
| `Qwen/Qwen3-VL-Embedding-2B` | Recommended base/stretch | Strong official Sentence Transformers example; use GPU, low batch size, and cached embeddings. |
| `BidirLM/BidirLM-Omni-2.5B-Embedding` | Stretch | Multimodal/omni embedding model; use if runtime and dependencies allow. |
| `LCO-Embedding/LCO-Embedding-Omni-3B` | Stretch | Larger multimodal option; useful for stronger submissions. |

The main run should use a model that directly consumes a combined image+text input and returns one vector. Avoid older dual-encoder designs whose final representation is made by manual text/image score fusion or external vector averaging.

Example interface:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

product_inputs = [
    {"text": row.product_text, "image": row.image_path}
    for row in catalog.itertuples()
]

product_dense = model.encode_document(
    product_inputs,
    batch_size=2,
    normalize_embeddings=True,
    show_progress_bar=True,
)
```

If the chosen model does not implement `encode_document`, use `encode` with the same multimodal input objects and document this in `index_manifest.json`.

## 6.4 Dense Vector Requirements

- Store exactly one dense vector per product.
- The vector dimension must be the model output dimension.
- Normalize vectors if using cosine similarity or dot product over normalized vectors.
- Cache vectors to disk. Do not recompute embeddings in every retrieval run.
- Store product IDs in the same order as the dense matrix rows.

Required dense artifact relationship:

```text
product_dense.npy row i  <->  product_ids.txt line i
```

## 6.5 Sparse Product Embedding

Use a SPLADE++-style sparse encoder over `product_text`. Acceptable models include:

- `naver/splade-cocondenser-ensembledistil`
- `naver/splade-v3`
- `prithivida/Splade_PP_en_v1`
- `prithivida/Splade_PP_en_v2`
- another SPLADE-like model approved by the instructor

The sparse vector should represent weighted vocabulary dimensions. Store it efficiently as token IDs and weights, not as a dense 30k-dimensional array per product.

Recommended sparse vector format:

```json
{
  "product_id": "B075X4QMX3|amazon.com",
  "indices": [2156, 3241, 9812],
  "values": [1.34, 0.91, 0.44]
}
```

To keep memory manageable, keep only the top 128 or top 256 non-zero sparse dimensions per product.

## 6.6 Index Options

### Option A: Qdrant Local

Use Qdrant local mode or a local Qdrant server with two vector fields:

- `dense`: single-vector multimodal dense vector;
- `sparse`: SPLADE++-style sparse vector.

This is the cleanest option for hybrid retrieval because dense and sparse vectors can live in one collection.

### Option B: FAISS + Sparse Matrix

Use:

- FAISS for dense vectors;
- SciPy CSR matrix, Pyserini, or another sparse search implementation for sparse vectors;
- a product ID mapping file to merge results.

This option is often simpler in Colab/Kaggle if installing a vector database is inconvenient.

## 6.7 Index Manifest

Create:

```text
artifacts/index/index_manifest.json
```

Required fields:

```json
{
  "num_products": 2000,
  "dense_model": "Qwen/Qwen3-VL-Embedding-2B",
  "dense_input_format": "multimodal_dict_text_image",
  "dense_embedding_policy": "single_vector_from_combined_text_image_input",
  "manual_text_image_vector_mixing": false,
  "sparse_model": "naver/splade-v3",
  "dense_dimension": 2048,
  "sparse_top_n": 128,
  "vector_store": "faiss_plus_csr",
  "created_at": "2026-06-10T00:00:00Z"
}
```

## 6.8 Indexing Summary

`reports/indexing_summary.md` must include:

- dense model name and loading configuration;
- sparse model name and loading configuration;
- dense vector dimension;
- dense batch size and runtime environment;
- number of product vectors produced;
- sparse top-n setting;
- average sparse non-zero count;
- index type and similarity function;
- at least three sanity-check searches.

## 6.9 Acceptance Tests

Stage 1 passes basic checks if:

- every product in `catalog_subset.parquet` has exactly one dense vector;
- dense vectors are generated from combined text+image product inputs;
- dense vector norms are close to 1.0 when normalization is enabled;
- every product has a sparse vector or a logged encoding failure;
- average sparse non-zero count is neither zero nor the full vocabulary;
- querying the dense index with a product's own image returns that product near the top;
- querying the dense index with a product's own text returns that product near the top;
- querying the sparse index with a product title returns that product near the top;
- product IDs in embeddings match product IDs in the catalog.

## 6.10 Common Mistakes

- Producing separate text and image vectors and averaging them as the final dense representation.
- Using image-only dense embeddings and calling them multimodal.
- Using text-only dense embeddings and ignoring product images.
- Storing SPLADE output as dense arrays, causing unnecessary memory use.
- Forgetting to cache embeddings.
- Mismatching product ID order between dense vectors and catalog rows.

---

# Stage 2 — Text, Image, and Image+Text Candidate Retrieval

## 7. Goal

Given a user query, retrieve a candidate set of relevant products using:

- single-vector multimodal dense retrieval;
- sparse retrieval when the query contains text;
- a prefusion method that combines dense and sparse evidence before cross-encoder reranking.

## 7.1 Required Input

```text
data/catalog_subset.parquet
artifacts/index/*
data/queries.jsonl
```

## 7.2 Query Schema

Each query must be one JSON object per line.

Text query:

```json
{
  "query_id": "q001",
  "query_text": "gray fabric sofa for a small living room, not leather",
  "query_image_path": null,
  "query_type": "text",
  "split": "test"
}
```

Image query:

```json
{
  "query_id": "q002",
  "query_text": null,
  "query_image_path": "data/query_images/example_chair.jpg",
  "query_type": "image",
  "split": "test"
}
```

Image+text query:

```json
{
  "query_id": "q003",
  "query_text": "similar style, but in black and not a dining chair",
  "query_image_path": "data/query_images/reference_chair.jpg",
  "query_type": "image_text",
  "split": "test"
}
```

Validation rules:

- `query_text` must be non-empty for `text` and `image_text` queries.
- `query_image_path` must be non-empty for `image` and `image_text` queries.
- At least one of `query_text` or `query_image_path` must be non-empty.
- `query_type` must be one of `text`, `image`, or `image_text`.

## 7.3 Required Output

```text
outputs/retrieval_results.jsonl
```

Each line:

```json
{
  "query_id": "q001",
  "query_type": "text",
  "run_name": "hybrid_dense_sparse_prefusion",
  "sparse_applicable": true,
  "results": [
    {
      "rank": 1,
      "product_id": "B075X4QMX3|amazon.com",
      "dense_score": 0.71,
      "sparse_score": 13.42,
      "prefusion_score": 0.88,
      "source": ["dense", "sparse"]
    }
  ]
}
```

For image-only queries, use:

```json
"sparse_applicable": false
```

and set `sparse_score` to `null`.

## 7.4 Dense Query Encoding

Use the same dense model and same input policy as Stage 1.

Text-only query:

```python
query_input = query_text
query_dense = model.encode_query(query_input, normalize_embeddings=True)
```

Image-only query:

```python
query_input = query_image_path
query_dense = model.encode_query(query_input, normalize_embeddings=True)
```

Image+text query:

```python
query_input = {"text": query_text, "image": query_image_path}
query_dense = model.encode_query(query_input, normalize_embeddings=True)
```

The image+text query must produce one vector directly from the combined multimodal input object. Do not encode the text and image separately and manually combine them.

## 7.5 Sparse Query Encoding

For queries with text, encode `query_text` using the same sparse model used for products.

Sparse retrieval is required for:

- text queries;
- image+text queries.

Sparse retrieval is not applicable for pure image-only queries unless the system explicitly creates a query caption. If a caption is used, it must be logged in the retrieval output or summary report.

## 7.6 Candidate Generation

For each query:

1. Encode the query with the multimodal dense model.
2. Retrieve `top_k_dense` candidates from the dense index.
3. If `query_text` exists, encode it with the sparse model.
4. Retrieve `top_k_sparse` candidates from the sparse index.
5. Merge candidates by `product_id`.
6. Keep source information: dense only, sparse only, or both.
7. Compute a prefusion score for the non-reranked hybrid baseline.

Recommended defaults:

```text
top_k_dense = 100
top_k_sparse = 100
final_candidate_pool = up to 200 unique products
```

## 7.7 Prefusion Baseline

Before cross-encoder reranking, implement at least one hybrid prefusion method.

### Reciprocal Rank Fusion

```text
RRF(product) = sum over retrieval systems 1 / (k + rank_system(product))
```

Recommended:

```text
k = 60
```

### Weighted Normalized Score Fusion

```text
prefusion_score = w_dense * normalized_dense_score
                + w_sparse * normalized_sparse_score
```

Recommended defaults for text and image+text queries:

```text
w_dense = 0.55
w_sparse = 0.45
```

For image-only queries:

```text
prefusion_score = normalized_dense_score
```

because sparse evidence is missing.

## 7.8 Required Query Set

Create at least:

- 10 text-only queries;
- 5 image-only queries;
- 5 image+text queries.

At least five queries must include a difficult constraint, such as:

- negation: `not leather`, `not a dining chair`, `not a table lamp`;
- exact attribute: color, material, style, product type;
- use case: `for a small living room`, `for office`, `for bedside storage`;
- visual reference plus textual change: `similar shape but darker`, `same style but wooden`.

## 7.9 Retrieval Summary

Create:

```text
reports/retrieval_summary.md
```

Include:

- number of queries by type;
- dense top-K and sparse top-K;
- prefusion method and weights;
- average number of unique candidates per query;
- number of candidates appearing in both dense and sparse results;
- examples of top results for at least three queries.

## 7.10 Acceptance Tests

Stage 2 passes basic checks if:

- every query returns at most the requested number of candidates;
- no duplicate `product_id` appears in one query's result list;
- results are sorted by rank;
- text queries use both dense and sparse retrieval;
- image+text queries use both dense and sparse retrieval;
- image-only queries do not crash when sparse retrieval is not applicable;
- output schema is stable and can be consumed by the reranking stage.

## 7.11 Common Mistakes

- Encoding image+text queries as two vectors and manually averaging them.
- Forgetting sparse retrieval for image+text queries.
- Returning duplicate products after merging dense and sparse results.
- Comparing raw dense and sparse scores without normalization.
- Evaluating only easy text queries and ignoring image or image+text queries.

---

# Stage 3 — Cross-Encoder Fusion and Reranking

## 8. Goal

Improve ranking quality by applying a cross-encoder or multimodal reranker to the candidate products returned by Stage 2. The reranker should better handle fine-grained matching, visual-text alignment, product attributes, and negative constraints such as `not leather` or `not a table lamp`.

## 8.1 Required Input

```text
outputs/retrieval_results.jsonl
data/catalog_subset.parquet
data/queries.jsonl
```

## 8.2 Required Output

```text
outputs/reranked_results.jsonl
reports/reranking_summary.md
```

Output line example:

```json
{
  "query_id": "q001",
  "run_name": "hybrid_cross_encoder_fusion",
  "reranker_type": "multimodal_cross_encoder",
  "results": [
    {
      "rank": 1,
      "product_id": "B075X4QMX3|amazon.com",
      "dense_score": 0.71,
      "sparse_score": 13.42,
      "prefusion_score": 0.88,
      "cross_encoder_score": 8.93,
      "final_score": 0.94,
      "source": ["dense", "sparse"]
    }
  ]
}
```

## 8.3 Reranker Model Choices

The recommended reranking approach is a multimodal cross-encoder that can score a query object against a product object where either side may contain text, image, or both.

Recommended models:

| Model | Suggested role | Notes |
|---|---|---|
| `jinaai/jina-reranker-m0` | Base multimodal reranker | Supports text-image pair scoring in the Sentence Transformers interface. |
| `nvidia/llama-nemotron-rerank-vl-1b-v2` | Stronger multimodal reranker | Good stretch choice if GPU memory allows. |
| `Qwen/Qwen3-VL-Reranker-2B` | High-quality multimodal reranker | More demanding; batch carefully. |

A text-only cross-encoder may be used as an ablation, but the main multimodal project run should preserve visual evidence through either a multimodal reranker or score fusion with the dense multimodal score.

## 8.4 Reranker Pair Construction

Construct the query object according to query type:

Text query:

```python
rerank_query = query_text
```

Image query:

```python
rerank_query = query_image_path
```

Image+text query:

```python
rerank_query = {"text": query_text, "image": query_image_path}
```

Construct each product object as:

```python
rerank_product = {
    "text": product_text,
    "image": product_image_path,
}
```

The reranker receives pairs like:

```python
(rerank_query, rerank_product)
```

or uses a `rank(query, documents)` method with the same query and product objects.

## 8.5 Text for Reranking

The `product_text` sent to the reranker should include:

- title;
- product type;
- category;
- brand if available;
- color;
- material;
- style;
- useful bullet points;
- short description.

Keep the text short enough for the reranker maximum length. Do not include hidden relevance labels, ground-truth judgments, or evaluation notes.

## 8.6 Fusion with Dense and Sparse Evidence

The final project must compare at least two reranking/fusion strategies.

### Strategy A — Cross-Encoder Only Ranking

```text
final_score = cross_encoder_score
```

This tests whether the reranker alone improves candidate ordering.

### Strategy B — Cross-Encoder + Retrieval Evidence

Normalize dense, sparse, and cross-encoder scores per query, then combine:

```text
final_score = w_ce     * z(cross_encoder_score)
            + w_dense  * z(dense_score)
            + w_sparse * z(sparse_score)
```

Recommended defaults:

| Query type | `w_ce` | `w_dense` | `w_sparse` |
|---|---:|---:|---:|
| text | 0.65 | 0.20 | 0.15 |
| image | 0.60 | 0.40 | 0.00 |
| image_text | 0.55 | 0.30 | 0.15 |

If a score source is not applicable, set it to `null` and renormalize the remaining weights or use the table above with zero weight.

## 8.7 Reranking Depth

Recommended setting:

```text
rerank_top_n = 100
```

For very small GPUs, use:

```text
rerank_top_n = 50
```

For stronger experiments:

```text
rerank_top_n = 200
```

Do not apply the reranker to the full corpus. Reranking is intended for a candidate pool produced by fast dense and sparse retrieval.

## 8.8 Reranking Summary

`reports/reranking_summary.md` must include:

- reranker model name;
- reranker input format;
- reranking depth;
- score normalization method;
- fusion weights;
- runtime environment;
- at least three before/after examples;
- one example involving negation;
- one example involving image+text search.

## 8.9 Acceptance Tests

Stage 3 passes basic checks if:

- reranking does not introduce product IDs that were not in the Stage 2 candidate pool;
- output has the same `query_id` values as retrieval input;
- ranks are consecutive integers starting from 1;
- `cross_encoder_score` is present for candidates reranked by the selected reranker;
- `final_score` is present for every returned product;
- visual evidence is not discarded for image and image+text queries;
- at least one query with a negative constraint is improved or carefully analyzed.

## 8.10 Common Mistakes

- Applying the reranker to the whole catalog.
- Dropping the product image before multimodal reranking.
- Dropping dense visual evidence when using a text-only ablation.
- Comparing raw dense, sparse, and reranker scores without normalization.
- Using only the product title when useful evidence appears in color, material, category, or bullet points.
- Claiming reranking improves recall when the candidate pool has not changed. Reranking usually improves Precision@K and NDCG@K, while Recall@large-K may stay the same.

---

# Stage 4 — Evaluation

## 9. Goal

Measure retrieval quality before and after reranking and determine whether hybrid retrieval and cross-encoder fusion improve the system.

## 9.1 Required Input

```text
data/queries.jsonl
data/qrels.jsonl
outputs/retrieval_results.jsonl
outputs/reranked_results.jsonl
```

## 9.2 Required Output

```text
reports/evaluation_report.md
reports/evaluation_metrics.csv
reports/error_analysis.md
```

## 9.3 Relevance Judgments

Create:

```text
data/qrels.jsonl
```

Each line:

```json
{
  "query_id": "q001",
  "product_id": "B075X4QMX3|amazon.com",
  "relevance": 2,
  "reason": "Gray fabric sofa; suitable for living room; no leather evidence."
}
```

Use graded relevance:

| Relevance | Meaning |
|---:|---|
| 2 | Strong match / exact match |
| 1 | Partial match or useful substitute |
| 0 | Not relevant |

## 9.4 Pooling Method for qrels

For each query, pool products from multiple runs:

1. dense-only single-vector multimodal retrieval;
2. sparse-only retrieval for text-bearing queries;
3. hybrid prefusion retrieval;
4. cross-encoder reranked results.

Label at least the top 10-20 unique pooled products per query.

Judgment rules:

- Read the query carefully, including negative constraints.
- Use product text and product image.
- Mark unknown attributes as unknown rather than assuming them.
- A product that violates an explicit negative constraint should usually be `0` or `1`, not `2`.
- If the query is image-based, judge visual similarity and any text constraints together.

## 9.5 Metrics

Compute at least:

- Precision@5 and Precision@10;
- Recall@10 and Recall@20;
- MAP;
- NDCG@5 and NDCG@10;
- judged coverage: percentage of returned products that have qrels.

For graded metrics such as NDCG, use relevance values 0, 1, and 2.

## 9.6 Required Comparison Table

`reports/evaluation_metrics.csv` must contain at least these columns:

```text
run_name, query_type, num_queries, precision_at_5, precision_at_10,
recall_at_10, recall_at_20, map, ndcg_at_5, ndcg_at_10, judged_coverage
```

The report must include a table like:

| Run | Query type | P@10 | R@20 | MAP | NDCG@10 | Notes |
|---|---|---:|---:|---:|---:|---|
| Dense single-vector multimodal | all | | | | | image+text product vectors |
| Sparse SPLADE++ | text/image_text | | | | | text-bearing queries only |
| Hybrid prefusion | text/image_text | | | | | dense + sparse |
| Hybrid + cross-encoder | all | | | | | final required run |
| Hybrid + cross-encoder + retrieval-score fusion | all | | | | | final fused run |

For image-only queries, report dense and multimodal-reranker performance separately from sparse retrieval because sparse retrieval has no original text query.

## 9.7 Error Analysis

`reports/error_analysis.md` must include at least five qualitative examples:

- one case where dense retrieval works better than sparse retrieval;
- one case where sparse retrieval works better than dense retrieval;
- one case where cross-encoder reranking improves ranking;
- one case where cross-encoder reranking hurts ranking;
- one case involving an image-only or image+text query.

For each case, show:

- the query;
- top results before reranking;
- top results after reranking;
- relevance labels if available;
- a short explanation of what happened.

## 9.8 Acceptance Tests

Stage 4 passes basic checks if:

- metrics match a small hand-computed toy example;
- missing qrels are handled gracefully;
- judged coverage is reported;
- all systems are evaluated on the same query split;
- text, image, and image+text performance are reported separately;
- the report includes both metrics and qualitative analysis.

## 9.9 Common Mistakes

- Evaluating different systems on different query sets.
- Treating unjudged products as definitely irrelevant without reporting coverage.
- Averaging sparse-only results over image-only queries.
- Reporting only aggregate scores and no error analysis.
- Optimizing weights on the test split.

---

# Stage 5 — Structured Output with a Small Language Model

## 10. Goal

Convert the top reranked products into a structured JSON response that is useful for a product search user.

The language model must not replace retrieval. It should explain and structure the results produced by the retrieval and reranking system.

## 10.1 Required Input

```text
data/queries.jsonl
outputs/reranked_results.jsonl
data/catalog_subset.parquet
```

Use only the top 5-10 products for each query.

## 10.2 Required Output

```text
outputs/final_answers.jsonl
reports/llm_output_summary.md
```

Each line of `outputs/final_answers.jsonl` must be a valid JSON object.

## 10.3 Required JSON Output Schema

```json
{
  "query_id": "q001",
  "interpreted_need": {
    "category": "sofa",
    "use_case": "small living room",
    "positive_preferences": ["gray", "fabric"],
    "negative_constraints": ["not leather"],
    "visual_preferences": [],
    "uncertain_fields": []
  },
  "product_judgements": [
    {
      "product_id": "B075X4QMX3|amazon.com",
      "role": "exact",
      "evidence": ["sofa", "fabric", "living room"],
      "constraint_violations": [],
      "reason": "The product is a fabric sofa suitable for a living room and does not appear to violate the leather constraint."
    },
    {
      "product_id": "B000EXAMPLE|amazon.com",
      "role": "substitute",
      "evidence": ["sofa", "living room"],
      "constraint_violations": ["material is leather"],
      "reason": "It matches the sofa category but violates the user's negative constraint against leather."
    }
  ],
  "decision": "recommend_exact_with_warning",
  "customer_response": "The best match is the gray fabric sofa because it fits the living-room use case and avoids leather. I would not choose the leather sofa as the first option because you explicitly said not leather."
}
```

## 10.4 Allowed Product Roles

| Role | Meaning |
|---|---|
| `exact` | Strong match for the user need and constraints. |
| `substitute` | Reasonable alternative but misses or weakly satisfies at least one preference. |
| `irrelevant` | Does not meaningfully satisfy the need. |

## 10.5 Allowed Decision Labels

| Decision | Meaning |
|---|---|
| `recommend_exact` | At least one exact match exists and no important warning is needed. |
| `recommend_exact_with_warning` | Best item is good, but there are caveats or close substitutes. |
| `recommend_substitute` | No exact match exists, but one or more substitutes are useful. |
| `no_good_match` | Retrieved products are mostly irrelevant. |
| `ask_clarification` | The query is too ambiguous or the top results are conflicting. |

## 10.6 Model Choices

Use a small local instruction-following language model if possible. Acceptable examples include small models from the Qwen, Phi, Llama, Gemma, or similar families. A hosted model may be used only with instructor permission.

The model must be prompted to output JSON only. The JSON must be validated with code.

## 10.7 Prompting Requirements

The prompt must include:

- user query text if present;
- note that a query image was provided if present;
- top products with product IDs and selected metadata;
- allowed role labels;
- allowed decision labels;
- exact JSON schema or a clear schema description;
- instruction not to invent product attributes.

Recommended system instruction:

```text
You are a product search assistant. Return only valid JSON.
Use only the product metadata provided. Do not invent price, availability, reviews, materials, or stock status.
If a field is unknown, mark it as unknown or leave it out.
```

## 10.8 JSON Validation

Implement automatic validation:

- parse with `json.loads`;
- check required keys;
- check allowed role labels;
- check allowed decision labels;
- check that every judged `product_id` came from the reranked input;
- check that the response is not empty;
- retry or repair invalid JSON if necessary.

A simple retry loop is acceptable:

```text
try prompt -> parse JSON -> validate schema
if invalid, prompt again with the error message
maximum retries = 2
if still invalid, return a deterministic fallback JSON
```

## 10.9 LLM Output Summary

`reports/llm_output_summary.md` must include:

- language model name;
- prompting template;
- number of generated answers;
- JSON validity rate;
- schema validity rate;
- number of repaired outputs;
- three example final answers;
- at least one failure case or limitation.

## 10.10 Acceptance Tests

Stage 5 passes basic checks if:

- 100% of final outputs are valid JSON after repair;
- every product ID in `product_judgements` exists in the input top products;
- all roles and decisions are from the allowed label sets;
- the model does not invent unavailable attributes such as price or stock;
- the customer response is understandable and consistent with the judgments.

## 10.11 Common Mistakes

- Letting the model output Markdown instead of JSON.
- Asking the model to choose products not present in the reranked input.
- Inventing price, reviews, stock status, or materials.
- Ignoring negative constraints in the explanation.
- Producing a customer response that contradicts the product judgments.

---

# 11. Required Repository Structure

```text
project/
  phase3_hybrid_multimodal_product_search_assignment.md
  air_phase3_complete_todo.ipynb
  pyproject.toml
  data/
    catalog_subset.parquet
    prep_stats.json
    queries.jsonl
    AIR_Phase3_stage1_images.zip
    raw/images/small/...          # created after extracting the image zip
  artifacts/
    embeddings/
      product_dense.npy
      product_sparse.jsonl
      product_ids.txt
    index/
      dense_index.*
      sparse_index.*
      index_manifest.json
  outputs/
    retrieval_results.jsonl
    reranked_results.jsonl
    final_answers.jsonl
  reports/
    data_preparation_summary.md
    indexing_summary.md
    retrieval_summary.md
    reranking_summary.md
    evaluation_report.md
    evaluation_metrics.csv
    error_analysis.md
    llm_output_summary.md
  src/
    tests.py
    build_judgment_pool.py
    retrieve.py
    rerank.py
    evaluate.py
    generate_answer.py
```

`data/qrels.jsonl`, `data/qrels_pool.csv`, generated outputs, reports, indexes, embeddings, and query caches are created during the project. They are not part of the fixed input package.

---

# 12. Suggested Command-Line Pipeline

The project should be runnable in this order. Stage 1 is in the notebook because it downloads and runs the multimodal embedding model on a GPU.

```bash
unzip -q data/AIR_Phase3_stage1_images.zip -d .
uv run python src/tests.py --stage 0 --catalog data/catalog_subset.parquet

# Run the Stage 1 notebook cells to create artifacts/embeddings and artifacts/index.

uv run python src/retrieve.py \
  --catalog data/catalog_subset.parquet \
  --queries data/queries.jsonl \
  --index_dir artifacts/index \
  --emb_dir artifacts/embeddings \
  --top_k_dense 100 \
  --top_k_sparse 100 \
  --out outputs/retrieval_results.jsonl

uv run python src/rerank.py \
  --catalog data/catalog_subset.parquet \
  --queries data/queries.jsonl \
  --retrieval outputs/retrieval_results.jsonl \
  --reranker jinaai/jina-reranker-v3 \
  --rerank_top_n 100 \
  --out outputs/reranked_results.jsonl

uv run python src/build_judgment_pool.py --pool_out data/qrels_pool.csv
# Fill relevance and reason in data/qrels_pool.csv, then finalize it:
uv run python src/build_judgment_pool.py \
  --finalize data/qrels_pool.csv \
  --qrels_out data/qrels.jsonl

uv run python src/evaluate.py \
  --queries data/queries.jsonl \
  --qrels data/qrels.jsonl \
  --runs outputs/retrieval_results.jsonl outputs/reranked_results.jsonl \
  --out reports/evaluation_metrics.csv

uv run python src/generate_answer.py \
  --catalog data/catalog_subset.parquet \
  --queries data/queries.jsonl \
  --reranked outputs/reranked_results.jsonl \
  --out outputs/final_answers.jsonl
```

---

# 13. Colab/Kaggle Feasibility Guidelines

The base project is designed to be feasible on common notebook GPUs if the subset is kept small and all embeddings are cached.

Recommended limits:

| Component | Recommended base setting | Stretch setting |
|---|---:|---:|
| Number of products | 1,000-2,000 | 5,000 |
| Image size | ABO small images, max 256px | same |
| Dense model | `Qwen/Qwen3-VL-Embedding-2B` with small batch size, or `nvidia/llama-nemotron-embed-vl-1b-v2` | larger omni/VL embedding models if runtime allows |
| Sparse model | SPLADE-v3 or SPLADE++ distilled, top 128 terms | top 256 terms |
| Dense top-K | 100 | 200 |
| Sparse top-K | 100 | 200 |
| Rerank depth | 50-100 | 200 |
| Evaluation queries | 20-40 | 60+ |
| LLM input products | top 5-10 | top 10-20 |

Practical advice:

- Start with 200 products for debugging.
- Cache all dense and sparse embeddings to disk.
- Use batch inference for dense embeddings, sparse embeddings, and reranking.
- Do not run SPLADE repeatedly inside a query loop.
- Do not rerank more than 100-200 candidates per query unless runtime allows it.
- If GPU memory is limited, build dense embeddings, save them, restart the runtime, then build sparse embeddings.
- If a large sparse model is too slow, use a smaller SPLADE-like model and clearly report the tradeoff.
- Keep an environment log with model names, package versions, GPU type, and runtime.

The release uses `pyproject.toml` as the dependency source. Do not keep a separate requirements file for this project. Typical local setup:

```bash
uv sync
uv sync --extra stage1-gpu
```

For the local LLM section, use a separate GPU runtime and install the Stage 5 group:

```bash
uv sync --extra stage5-llm
```

---

# 14. Grading Rubric

| Component | Points | What is graded |
|---|---:|---|
| Data preparation | 12 | Correct ABO subset, clean schema, valid images, reproducibility. |
| Single-vector multimodal dense embedding | 14 | Combined text+image input, one vector per product, normalization, caching, correct query encoding. |
| Sparse SPLADE++ embedding | 10 | Correct sparse product/query vectors, efficient storage, usable sparse search. |
| Indexing and retrieval | 14 | Dense retrieval, sparse retrieval, hybrid candidate merge, text/image/image+text handling. |
| Cross-encoder fusion/reranking | 14 | Correct pair construction, rerank depth, score fusion, negative constraints, preservation of visual evidence. |
| Evaluation | 12 | qrels, P@K, Recall@K, MAP, NDCG, judged coverage, fair comparison. |
| Structured LLM output | 10 | Valid JSON, correct schema, grounded judgments, useful response. |
| Report and analysis | 10 | Architecture explanation, design decisions, tables, error analysis, limitations. |
| Code quality and reproducibility | 4 | Clear structure, rerunnable commands, saved artifacts. |
| Total | 100 |  |

Extra credit may be awarded for:

- fine-tuning the structured-output language model with SFT;
- using GRPO or another reinforcement-learning method to improve JSON validity or judgment quality;
- training or adapting a reranker on project-specific data;
- advanced hybrid retrieval analysis;
- strong visualization in the final demo;
- careful ablation of dense, sparse, and reranker components.

---

# 15. Required Final Report

The final report should be concise but complete. It must include:

1. dataset subset construction;
2. final number of products and product-type distribution;
3. dense model and input format;
4. sparse model and sparse-vector format;
5. vector database or index design;
6. query set design;
7. retrieval method;
8. cross-encoder reranking and fusion method;
9. evaluation metrics table;
10. error analysis;
11. examples of final structured JSON answers;
12. limitations and possible improvements.

Recommended report length: 6-10 pages, excluding appendix tables and code.

---

# 16. Required Query and qrels Files

## 16.1 Query File Schema

```json
{
  "query_id": "string, required",
  "query_text": "string or null, required",
  "query_image_path": "string or null, required",
  "query_type": "one of: text, image, image_text",
  "split": "train, dev, or test"
}
```

## 16.2 qrels Schema

```json
{
  "query_id": "string",
  "product_id": "string",
  "relevance": "integer 0, 1, or 2",
  "reason": "string"
}
```

Validation rules:

- no duplicate `(query_id, product_id)` pairs;
- relevance must be one of `0`, `1`, or `2`;
- every qrel query must exist in `queries.jsonl`;
- every qrel product must exist in `catalog_subset.parquet`.

---

# 17. Minimal Ranking File Schema

Both retrieval and reranking files must be JSONL files. Each line corresponds to one query.

```json
{
  "query_id": "string",
  "query_type": "text | image | image_text",
  "run_name": "string",
  "sparse_applicable": true,
  "results": [
    {
      "rank": 1,
      "product_id": "string",
      "dense_score": "number or null",
      "sparse_score": "number or null",
      "prefusion_score": "number or null",
      "cross_encoder_score": "number or null",
      "final_score": "number",
      "source": ["dense", "sparse"]
    }
  ]
}
```

Validation rules:

- ranks must start from 1 and be consecutive;
- product IDs must be unique per query;
- `final_score` must be present in reranked results;
- retrieval results may use `prefusion_score` as the final ordering score;
- `cross_encoder_score` is expected in reranked results for candidates scored by the selected reranker;
- `sparse_score` may be null for image-only queries.

---

# 18. Suggested Starter Query Set

Adapt these queries to the actual subset. Replace queries that have no possible relevant products.

```jsonl
{"query_id":"q001","query_text":"gray fabric sofa for a small living room, not leather","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q002","query_text":"modern wooden coffee table with storage","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q003","query_text":"black office chair with armrests, not a dining chair","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q004","query_text":"white bookshelf for a bedroom","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q005","query_text":"floor lamp for living room, not a table lamp","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q006","query_text":"round dining table for four people","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q007","query_text":"beige area rug with a simple design, not colorful","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q008","query_text":"small storage cabinet for entryway","query_image_path":null,"query_type":"text","split":"test"}
{"query_id":"q009","query_text":null,"query_image_path":"data/query_images/query_chair_01.jpg","query_type":"image","split":"test"}
{"query_id":"q010","query_text":"similar style, but darker and not for dining","query_image_path":"data/query_images/query_chair_01.jpg","query_type":"image_text","split":"test"}
```

For image queries, query images may be:

- held-out product images from the ABO subset;
- ABO products not included in the search corpus;
- student-provided reference images, if license and privacy constraints are acceptable.

If an ABO product image is used as a query image, it should not be identical to a product image in the indexed corpus unless the query is explicitly an image-duplicate sanity check.

---

# 19. Suggested Baselines and Ablations

At minimum, compare:

1. **Dense text-only query/product encoding with the same multimodal embedding model**: text input only.
2. **Dense single-vector multimodal product encoding**: required text+image product input.
3. **Sparse SPLADE++ retrieval**: text-bearing queries only.
4. **Hybrid prefusion**: dense + sparse using RRF or weighted score fusion.
5. **Hybrid + cross-encoder**: reranked candidate pool.
6. **Hybrid + cross-encoder + retrieval-score fusion**: final fused run.

Useful ablations:

- dense top-K = 50 vs 100 vs 200;
- sparse top-K = 50 vs 100 vs 200;
- rerank depth = 20 vs 50 vs 100;
- RRF vs weighted score fusion;
- text-only vs image-only vs image+text query performance;
- multimodal reranker vs text-only reranker ablation.

Do not tune hyperparameters on the test set. Use a small development query split.

---

# 20. Design Rationale

## 20.1 Why Single-Vector Multimodal Dense Embeddings?

Product metadata describes category, material, dimensions, style, and use case. Product images describe shape, color, visual style, and appearance. A single-vector multimodal embedding model can read text and image as one combined input and produce one vector that represents the product as a whole.

This is cleaner than manually mixing two separate vectors because the model is responsible for combining modalities internally. The project therefore requires direct image+text input to the dense model for product embeddings and image+text queries.

## 20.2 Why Sparse Embeddings?

Sparse retrieval is useful for exact words, rare attributes, product types, brand names, and explicit constraints. SPLADE-style models can also expand terms, making them more semantic than classic keyword matching while still preserving lexical behavior.

Dense and sparse retrieval often make different mistakes. Hybrid retrieval gives the reranker a better candidate pool.

## 20.3 Why Cross-Encoder Fusion?

Dense and sparse retrievers encode query and product independently. This is fast, but it can miss interactions between the query and product. A cross-encoder or multimodal reranker scores each query-product pair directly, making it better suited for details such as:

- `not leather` vs `genuine leather`;
- `floor lamp` vs `table lamp`;
- `chair with armrests` vs `armless chair`;
- `small living room` vs a very large sofa;
- an image reference plus a text modification such as `same style but darker`.

Because cross-encoders are slower, they should rerank only a candidate pool.

## 20.4 Why JSON Output?

A product search system should not only return a ranked list. It should also communicate why products match or fail. JSON output makes the final decision machine-readable and easier to test.

---

# 21. Quality Checklist

## Data

- Can the subset be reproduced with the same seed?
- Are all local image paths valid?
- Are titles and product text mostly English?
- Was the large original-image archive avoided?
- Does the subset contain multiple product types?

## Embeddings and Indexes

- Is there exactly one dense vector per product?
- Was the dense vector produced from a combined text+image input object?
- Are dense vectors normalized if cosine/dot-product similarity is used?
- Are sparse vectors efficiently stored?
- Can dense and sparse retrieval each return reasonable sanity-check results?

## Retrieval

- Does text search use both dense and sparse retrieval?
- Does image search work without text?
- Does image+text search use a combined multimodal dense query object and sparse text query?
- Are candidate sets merged without duplicates?
- Is prefusion computed consistently?

## Reranking

- Is the reranker applied only to candidates?
- Are product texts well constructed for reranking?
- Are product images available to the multimodal reranker?
- Are scores normalized before weighted fusion?
- Is visual evidence still considered for image+text queries?

## Evaluation

- Are qrels created by pooling multiple systems?
- Are metrics computed on the same query split?
- Is image-only evaluation separated where sparse retrieval is not applicable?
- Does error analysis include both successes and failures?

## LLM Output

- Is every output valid JSON?
- Are all product IDs copied exactly from input?
- Does the model avoid inventing price, stock, review, or material information?
- Is the final customer response consistent with product judgments?

---

# 22. Expected Difficulty and Feasibility Analysis

This project is ambitious but feasible if implemented carefully.

## Easy Parts

- Creating a small subset after understanding the metadata structure.
- Implementing a dense index with FAISS.
- Creating JSONL query and output files.
- Generating structured JSON output with a constrained prompt.

## Medium Parts

- Parsing multilingual ABO fields cleanly.
- Joining listing metadata with image metadata.
- Running a multimodal embedding model over 1,000-2,000 product images.
- Implementing sparse neural retrieval efficiently.
- Creating fair qrels without spending too much manual labeling time.
- Keeping file schemas stable across stages.

## Hard Parts

- Making image+text queries behave well.
- Preserving visual evidence through reranking and fusion.
- Reranking negative constraints reliably.
- Ensuring fusion does not over-trust one score source.
- Writing meaningful error analysis instead of only reporting metrics.
- Fine-tuning a language model, if attempted.

## Practical Scope Control

A strong base submission should focus on:

```text
2,000 products
20-40 queries
dense top-100 + sparse top-100
cross-encoder rerank top-100
LLM output for top-5 products
```

This scope is enough to demonstrate the full retrieval pipeline without making the project too heavy for Colab or Kaggle.

---

# 23. What a Good Final Demo Looks Like

A good demo should show the user query, top products, and final JSON answer.

Example demo layout:

```text
Query: gray fabric sofa for a small living room, not leather

Top 5 after hybrid retrieval:
1. Product A — dense=0.72 sparse=12.1 prefusion=0.91
2. Product B — dense=0.68 sparse=9.7 prefusion=0.83
...

Top 5 after cross-encoder fusion:
1. Product B — CE=8.9 final=0.94
2. Product A — CE=7.1 final=0.86
...

Structured answer:
{
  "interpreted_need": ...,
  "product_judgements": ...,
  "decision": ...,
  "customer_response": ...
}
```

For image queries, show the query image and top product images in the notebook or report.

---

# 24. Instructor-Facing Consistency Checks

These checks are useful for grading scripts and manual review.

## 24.1 File Existence

Required files:

```text
data/catalog_subset.parquet
artifacts/embeddings/product_dense.npy
artifacts/embeddings/product_ids.txt
artifacts/index/index_manifest.json
outputs/retrieval_results.jsonl
outputs/reranked_results.jsonl
outputs/final_answers.jsonl
reports/evaluation_metrics.csv
```

## 24.2 Dense Embedding Policy Check

`index_manifest.json` must include:

```json
{
  "dense_input_format": "multimodal_dict_text_image",
  "dense_embedding_policy": "single_vector_from_combined_text_image_input",
  "manual_text_image_vector_mixing": false
}
```

## 24.3 Ranking Schema Check

For each line in retrieval and reranking outputs:

- `query_id` must exist in `queries.jsonl`;
- product IDs must exist in `catalog_subset.parquet`;
- ranks must be consecutive;
- scores must be numeric or null according to query type;
- image-only queries must set `sparse_applicable = false` unless an explicit caption-based sparse query is documented.

## 24.4 LLM Schema Check

For each final answer:

- output must parse as JSON;
- all product judgments must reference top reranked products;
- roles and decisions must be from allowed label sets;
- no unavailable product facts should be invented.

---

# 25. References

- Amazon Berkeley Objects dataset: `https://amazon-berkeley-objects.s3.amazonaws.com/index.html`
- Hugging Face multimodal Sentence Transformers guide: `https://huggingface.co/blog/multimodal-sentence-transformers`
- Sentence Transformers documentation: `https://www.sbert.net/`
- Qdrant documentation: `https://qdrant.tech/documentation/`
- FAISS documentation: `https://faiss.ai/`

---

# 26. Final Advice

Build the project in small, testable pieces. The best submissions are not necessarily the ones with the largest model. They are the ones with clean data, clear interfaces, fair evaluation, careful reranking, and honest analysis of where the system succeeds or fails.

A simple, reproducible hybrid system with stable artifacts and meaningful evaluation is better than a complex system whose outputs cannot be reproduced.
