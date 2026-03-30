# Shared Data Contract

The first-wave projects in this repository share a common interaction-table convention so the ranking, candidate generation, and serving projects all tell one coherent systems story.

## Primary Dataset

- Preferred source: Criteo Sponsored Search Conversion Logs
- Fallback: MovieLens 20M with synthetic catalog and query metadata

## Expected Processed Table

Store the cleaned table at:

```text
data/processed/commerce_interactions.csv
```

Parquet is also supported by the loaders if you prefer:

```text
data/processed/commerce_interactions.parquet
```

## Required Columns

| Column | Type | Purpose |
| --- | --- | --- |
| `user_id` | string/int | User identity for ranking and retrieval |
| `product_id` | string/int | Item identity for ranking and retrieval |
| `session_id` | string/int | Grouping key for ranking metrics like NDCG@K |
| `click` | 0/1 | Primary ranking target |
| `conversion` | 0/1 | Optional higher-value target |
| `category_id` | string/int | Item/category context |
| `device_type` | string | Slice analysis and sparse features |
| `price` | float | Dense item feature |
| `discount` | float | Dense item feature |
| `query_length` | float | Dense context feature |
| `product_age_days` | float | Dense item freshness feature |
| `user_recency_days` | float | Dense user freshness feature |

## Optional Columns

- `event_ts`
- `brand_id`
- `query_id`
- `query_text`
- `country`

## Preprocessing Convention

1. Normalize column names to snake case.
2. Coerce identifier columns to strings or ints consistently.
3. Fill missing dense features with zero or a documented sentinel.
4. Preserve the raw `click` and `conversion` labels.
5. Keep one row per user-item impression or interaction.
6. Use `session_id` as the ranking group when available; otherwise derive a stable grouping key.

## Reuse Across Projects

- `ranking-systems/01-dcnv2-commerce-ranking/` uses the table directly for pointwise ranking.
- `recommendation-systems/01-two-tower-candidate-generation/` uses the same table to derive positive interactions and sampled negatives.
- `serving-systems/01-onnx-latency-lab/` benchmarks the ranking model exported from Project 1 using sample feature rows from the same schema.
