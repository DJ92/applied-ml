# DCN/DCN-V2 Commerce Ranking Lab

Production-shaped ranking project for commerce relevance with sparse features, dense features, embeddings, calibration analysis, and ablation workflows.

## What This Shows

- Deep ranking for recommendation and search-style problems
- Handling sparse+dense feature mixes with learned embeddings
- Comparing simple baselines against DCN and DCN-V2
- Evaluating not just AUC and log loss, but also NDCG, calibration, and latency

## Problem

Ranking commerce candidates is a multi-objective problem: maximize relevance and downstream value while respecting latency budgets, sparse feature realities, and the need to explain trade-offs. This project builds a public-safe approximation of that setting using an interaction table with user, item, session, and context features.

## Dataset And Feature Schema

- Primary dataset: Criteo Sponsored Search Conversion Logs
- Fallback dataset: MovieLens 20M plus synthetic product metadata
- Shared schema: [../../data/README.md](../../data/README.md)

Default features:

- Categorical: `user_id`, `product_id`, `category_id`, `device_type`
- Dense: `price`, `discount`, `query_length`, `product_age_days`, `user_recency_days`
- Crosses: `user_x_category`
- Target: `click`
- Group key: `session_id`

## Models And Baselines

- Logistic ranking baseline
- Shallow MLP baseline
- DCN
- DCN-V2

## Public Interfaces

```bash
python src/train.py --config configs/dcnv2.yaml
python src/evaluate.py --model checkpoints/dcnv2.pt
python src/ablate.py --feature-group sparse
```

## Key Results

This project produces comparison tables for:

- AUC
- Log loss
- NDCG@10
- Expected calibration error
- CPU inference latency
- Parameter count

## Architecture

```text
interaction table
  -> feature validation
  -> categorical encoding + crossed features
  -> train/val/test split by session
  -> model training
  -> evaluation and slice analysis
  -> checkpoint bundle for serving export
```

## Trade-offs

- Sparse feature richness vs model complexity
- Ranking quality vs calibration quality
- Crossed features vs memory footprint
- Better models vs serving latency

## Failure Modes

- Leakage from unstable group splits
- Sparse IDs with too little signal
- Overfitting on head users or head items
- Calibration drift even when ranking metrics look healthy

## What I Would Improve In Production

- Add richer feature crosses and hashed embeddings
- Integrate experiment tracking and online/offline metric alignment
- Add post-ranking business rules and diversity-aware re-ranking
- Expand slice analysis to more user and catalog cohorts

## Testing

```bash
pytest tests -q
```
