# Stage 3 - Reranking Summary

- Reranker: `BAAI/bge-reranker-v2-m3` (multimodal cross-encoder)
- Pair format: (query_text_or_dict, product_text) built from title/category/brand/color/material/style/bullets/description
- Reranking depth: 100
- Score normalization: per-query z-score for cross-encoder, dense, and sparse scores
- Fusion weights: {"text": {"ce": 0.65, "dense": 0.2, "sparse": 0.15}, "image": {"ce": 0.6, "dense": 0.4, "sparse": 0.0}, "image_text": {"ce": 0.55, "dense": 0.3, "sparse": 0.15}}

## Before/after examples

**q001**: before=['B086B5MRFB|amazon.in', 'B07HPNBTHF|amazon.co.uk', 'B07QJXW4JR|amazon.com', 'B07B4D88DY|amazon.com', 'B0723H8HJY|amazon.com'] | after=['B086B5MRFB|amazon.in', 'B07B4MRLT8|amazon.com', 'B07QJXW4JR|amazon.com', 'B07K2JGTDN|amazon.com', 'B073G8Q656|amazon.com']
**q002**: before=['B07QC861L7|amazon.com', 'B07K7NNS6N|amazon.co.uk', 'B07J1YW3YT|amazon.com', 'B07QJLWJJM|amazon.ca', 'B07GF51SLB|amazon.co.uk'] | after=['B07DBF3VJY|amazon.com', 'B07K7NNS6N|amazon.co.uk', 'B07J1YW3YT|amazon.com', 'B07K7K4CM2|amazon.co.uk', 'B07SSCJKLN|amazon.in']
**q003**: before=['B07SSHYD2Z|amazon.co.uk', 'B081HWKWN6|amazon.co.uk', 'B07QLGRVPN|amazon.in', 'B081HX9SRM|amazon.co.uk', 'B075Z6S94K|amazon.com'] | after=['B07SSHYD2Z|amazon.co.uk', 'B081HX9SRM|amazon.co.uk', 'B081HWKWN6|amazon.co.uk', 'B07QLGRVPN|amazon.in', 'B07HMPPZRN|amazon.com']

## Negation example
Query q001: gray fabric sofa for a small living room, not leather
Top after rerank: ['B086B5MRFB|amazon.in', 'B07B4MRLT8|amazon.com', 'B07QJXW4JR|amazon.com', 'B07K2JGTDN|amazon.com', 'B073G8Q656|amazon.com']

## Image+text example
Query q016: similar style, but in black and not for dining (image: data/raw/images/small/3e/3ed459ef.jpg)
Top after rerank: ['B07SVJB925|amazon.co.uk', 'B07SGQ2W3J|amazon.co.uk', 'B07Q2FQL52|amazon.ca', 'B07M5M66HT|amazon.com', 'B07QFWGPL3|amazon.co.uk']
