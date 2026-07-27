# Stage 4 — Evaluation Report

Judged 20 of 20 queries via a pooled qrels file (data/qrels.jsonl). Metrics: Precision@5/10, Recall@10/20, MAP, NDCG@5/10, and judged coverage (share of top-20 results that have a qrel label).

## Comparison table

| run_name                   | query_type   |   num_queries |   precision_at_5 |   precision_at_10 |   recall_at_10 |   recall_at_20 |    map |   ndcg_at_5 |   ndcg_at_10 |   judged_coverage |
|:---------------------------|:-------------|--------------:|-----------------:|------------------:|---------------:|---------------:|-------:|------------:|-------------:|------------------:|
| dense_single_vector        | text         |            10 |           0.76   |             0.64  |         0.6444 |         0.9255 | 0.7306 |      0.7616 |       0.7523 |            0.785  |
| dense_single_vector        | all          |            20 |           0.8    |             0.67  |         0.6662 |         0.9471 | 0.7654 |      0.7383 |       0.7386 |            0.775  |
| dense_single_vector        | image        |             5 |           0.88   |             0.7   |         0.6962 |         0.9608 | 0.8369 |      0.75   |       0.7532 |            0.78   |
| dense_single_vector        | image_text   |             5 |           0.8    |             0.7   |         0.6798 |         0.9765 | 0.7636 |      0.6799 |       0.6966 |            0.75   |
| sparse_splade              | text         |            10 |           0.6    |             0.55  |         0.5062 |         0.6622 | 0.5444 |      0.5357 |       0.5744 |            0.74   |
| sparse_splade              | all          |            15 |           0.4267 |             0.38  |         0.3453 |         0.4493 | 0.3708 |      0.3694 |       0.3913 |            0.5767 |
| sparse_splade              | image_text   |             5 |           0.08   |             0.04  |         0.0235 |         0.0235 | 0.0237 |      0.0369 |       0.025  |            0.25   |
| hybrid_prefusion           | text         |            10 |           0.72   |             0.59  |         0.5482 |         0.7757 | 0.6743 |      0.7003 |       0.6815 |            0.73   |
| hybrid_prefusion           | all          |            20 |           0.68   |             0.555 |         0.5288 |         0.7812 | 0.6454 |      0.6018 |       0.6029 |            0.72   |
| hybrid_prefusion           | image        |             5 |           0.88   |             0.7   |         0.6962 |         0.9608 | 0.8369 |      0.75   |       0.7532 |            0.78   |
| hybrid_prefusion           | image_text   |             5 |           0.4    |             0.34  |         0.3228 |         0.6124 | 0.3963 |      0.2566 |       0.2952 |            0.64   |
| hybrid_cross_encoder       | text         |            10 |           0.68   |             0.54  |         0.526  |         0.7313 | 0.6128 |      0.6258 |       0.6424 |            0.62   |
| hybrid_cross_encoder       | all          |            15 |           0.48   |             0.38  |         0.3646 |         0.5175 | 0.445  |      0.4285 |       0.4389 |            0.4833 |
| hybrid_cross_encoder       | image_text   |             5 |           0.08   |             0.06  |         0.0417 |         0.0898 | 0.1093 |      0.0339 |       0.0318 |            0.21   |
| hybrid_cross_encoder_fused | text         |            10 |           0.7    |             0.56  |         0.5403 |         0.7531 | 0.6476 |      0.6533 |       0.6589 |            0.67   |
| hybrid_cross_encoder_fused | all          |            20 |           0.47   |             0.425 |         0.4021 |         0.5887 | 0.4682 |      0.3854 |       0.4391 |            0.5725 |
| hybrid_cross_encoder_fused | image        |             5 |           0.4    |             0.48  |         0.4562 |         0.7287 | 0.4185 |      0.2024 |       0.3832 |            0.71   |
| hybrid_cross_encoder_fused | image_text   |             5 |           0.08   |             0.1   |         0.0717 |         0.1198 | 0.159  |      0.0323 |       0.0553 |            0.24   |

## Limitations

- Metrics rely on pooled qrels (top 10-20 unique products per query across systems); unjudged products below the pool depth are treated as not relevant, so recall is a lower bound.
- Image-only queries have no sparse_splade or text-only cross-encoder ablation rows, since there is no query text to drive sparse retrieval.
