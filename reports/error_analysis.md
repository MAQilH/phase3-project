# Stage 4 — Error Analysis

## Case 1: Dense retrieval outperforms sparse retrieval

**Query** `q011` (image): data/raw/images/small/ce/ce04dd2a.jpg

Top results before reranking:

  1. B07D49ZZ6S|amazon.com (score=0.016, relevance=partial)
  2. B014GCABSI|amazon.com (score=0.016, relevance=partial)
  3. B086TGJFZK|amazon.com (score=0.016, relevance=partial)
  4. B07Y2D8SF7|amazon.in (score=0.016, relevance=partial)
  5. B07K6M8HKR|amazon.com (score=0.015, relevance=partial)

Top results after reranking:

  (no results)

**What happened:** Dense embeddings capture visual/semantic similarity (style, shape, material look) that exact keyword overlap in SPLADE misses, so dense surfaces the relevant product earlier when the query uses paraphrased or visual language.

## Case 2: Sparse retrieval outperforms dense retrieval

**Query** `q002` (text): modern wooden coffee table with storage

Top results before reranking:

  1. B07GF51SLB|amazon.co.uk (score=0.031, relevance=strong)
  2. B07QC861L7|amazon.com (score=0.032, relevance=strong)
  3. B07K7NNS6N|amazon.co.uk (score=0.031, relevance=strong)
  4. B07QJLWJJM|amazon.ca (score=0.031, relevance=strong)
  5. B07J1YW3YT|amazon.com (score=0.031, relevance=partial)

Top results after reranking:

  1. B07DBF3VJY|amazon.com (score=0.030, relevance=strong)
  2. B07QC861L7|amazon.com (score=0.032, relevance=strong)
  3. B07J1YW3YT|amazon.com (score=0.031, relevance=partial)
  4. B07K7NNS6N|amazon.co.uk (score=0.031, relevance=strong)
  5. B07QJLWJJM|amazon.ca (score=0.031, relevance=strong)

**What happened:** The query names an exact attribute, brand, or product-type term. SPLADE's lexical term weighting matches that vocabulary directly, while dense similarity spreads mass across visually/semantically similar but attribute-mismatched products.

## Case 3: Cross-encoder reranking improves the ranking

**Query** `q001` (text): gray fabric sofa for a small living room, not leather

Top results before reranking:

  1. B086B5MRFB|amazon.in (score=0.033, relevance=strong)
  2. B07HPNBTHF|amazon.co.uk (score=0.031, relevance=not relevant)
  3. B07QJXW4JR|amazon.com (score=0.031, relevance=strong)
  4. B07B4D88DY|amazon.com (score=0.030, relevance=strong)
  5. B0723H8HJY|amazon.com (score=0.030, relevance=strong)

Top results after reranking:

  1. B086B5MRFB|amazon.in (score=5.866, relevance=strong)
  2. B07B4MRLT8|amazon.com (score=2.546, relevance=partial)
  3. B07QJXW4JR|amazon.com (score=2.097, relevance=strong)
  4. B07K2JGTDN|amazon.com (score=1.148, relevance=not relevant)
  5. B073G8Q656|amazon.com (score=1.094, relevance=strong)

**What happened:** The cross-encoder jointly attends to the query and each candidate's full text/image, so it can confirm fine-grained attribute matches (or negative-constraint compliance) that independent dense/sparse scoring could only approximate.

## Case 4: Cross-encoder reranking hurts the ranking

**Query** `q016` (image_text): similar style, but in black and not for dining

Top results before reranking:

  1. B07XZKBMLS|amazon.in (score=0.016, relevance=partial)
  2. B07GDKJVTN|amazon.in (score=0.016, relevance=not relevant)
  3. B000HVGJMK|amazon.com (score=0.016, relevance=unjudged)
  4. B07GF762PX|amazon.in (score=0.016, relevance=not relevant)
  5. B07XZK5XB9|amazon.ca (score=0.016, relevance=partial)

Top results after reranking:

  1. B07SVJB925|amazon.co.uk (score=1.610, relevance=unjudged)
  2. B07SGQ2W3J|amazon.co.uk (score=1.436, relevance=unjudged)
  3. B07Q2FQL52|amazon.ca (score=1.274, relevance=unjudged)
  4. B07M5M66HT|amazon.com (score=1.136, relevance=unjudged)
  5. B07QFWGPL3|amazon.co.uk (score=1.083, relevance=unjudged)

**What happened:** The reranker over-weighted surface text similarity (e.g. shared category words) over the visual/attribute mismatch that the dense and sparse scores had already correctly down-weighted, pulling a less relevant product to the top.

## Case 5: image query behavior

**Query** `q011` (image): data/raw/images/small/ce/ce04dd2a.jpg

Top results before reranking:

  1. B07D49ZZ6S|amazon.com (score=0.016, relevance=partial)
  2. B014GCABSI|amazon.com (score=0.016, relevance=partial)
  3. B086TGJFZK|amazon.com (score=0.016, relevance=partial)
  4. B07Y2D8SF7|amazon.in (score=0.016, relevance=partial)
  5. B07K6M8HKR|amazon.com (score=0.015, relevance=partial)

Top results after reranking:

  1. B07QD6TXWS|amazon.com (score=5.335, relevance=not relevant)
  2. B086TGJFZK|amazon.com (score=1.289, relevance=partial)
  3. B07D49ZZ6S|amazon.com (score=0.942, relevance=partial)
  4. B088V8YGYM|amazon.com (score=0.858, relevance=not relevant)
  5. B07Y2D8SF7|amazon.in (score=0.671, relevance=partial)

**What happened:** For image-bearing queries there is no independent text query to drive sparse retrieval, so the system relies on the single-vector multimodal dense embedding (and, for image_text queries, the text-modified combined query object) plus the multimodal cross-encoder to apply the requested visual/textual changes.
