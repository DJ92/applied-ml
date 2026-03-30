# Shared Data Contracts

This repository now carries two public-safe dataset contracts:

- a commerce interaction table for ranking, candidate generation, and serving
- a payment-events table for fraud and risk decisioning

## Commerce Dataset

- Preferred source: Criteo Sponsored Search Conversion Logs
- Fallback: MovieLens 20M with synthetic catalog and query metadata

### Expected Processed Table

Store the cleaned table at:

```text
data/processed/commerce_interactions.csv
```

Parquet is also supported by the loaders if you prefer:

```text
data/processed/commerce_interactions.parquet
```

### Required Columns

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

### Optional Columns

- `event_ts`
- `brand_id`
- `query_id`
- `query_text`
- `country`

### Preprocessing Convention

1. Normalize column names to snake case.
2. Coerce identifier columns to strings or ints consistently.
3. Fill missing dense features with zero or a documented sentinel.
4. Preserve the raw `click` and `conversion` labels.
5. Keep one row per user-item impression or interaction.
6. Use `session_id` as the ranking group when available; otherwise derive a stable grouping key.

### Reuse Across Projects

- `ranking-systems/01-dcnv2-commerce-ranking/` uses the table directly for pointwise ranking.
- `recommendation-systems/01-two-tower-candidate-generation/` uses the same table to derive positive interactions and sampled negatives.
- `serving-systems/01-onnx-latency-lab/` benchmarks the ranking model exported from Project 1 using sample feature rows from the same schema.

## Payment Risk Dataset

- Preferred source: PaySim
- Goal: public-safe stand-in for payment fraud and risk flows

### Expected Processed Table

Store the cleaned table at:

```text
data/processed/payment_risk_events.csv
```

### Required Columns

| Column | Type | Purpose |
| --- | --- | --- |
| `step` | int | Event-time ordering key |
| `type` | string | Transaction context |
| `amount` | float | Payment magnitude |
| `customer_id` | string | Originating account |
| `destination_id` | string | Destination account or merchant |
| `is_fraud` | 0/1 | Fraud label |
| `split` | string | Train / val / test routing |
| `hour_of_day` | float | Time-of-day feature |
| `day_index` | float | Coarser event-time feature |
| `customer_prior_txn_count` | float | Origin-account activity signal |
| `customer_avg_amount` | float | Origin-account behavior baseline |
| `customer_prior_fraud_rate` | float | Historical risk signal |
| `destination_prior_txn_count` | float | Destination activity signal |
| `destination_avg_amount` | float | Destination behavior baseline |
| `hours_since_prev_customer_txn` | float | Velocity proxy |
| `origin_balance_gap` | float | Balance anomaly signal |
| `destination_balance_gap` | float | Counterparty anomaly signal |
| `is_flagged_fraud_signal` | float | Upstream rule signal |

### Preprocessing Convention

1. Preserve causal ordering using `step`.
2. Derive historical aggregates using only prior events.
3. Split by time, not randomly.
4. Keep raw labels separate from operational decisions such as approve / review / decline.
5. Preserve enough context to simulate both scoring and manual-review queue behavior.

### Reuse Across Projects

- `risk-systems/01-realtime-payment-risk-decisioning/` uses this table for deterministic prep, training, stream scoring, evaluation, and review-queue simulation.
