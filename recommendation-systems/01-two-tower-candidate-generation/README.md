# Two-Tower Candidate Generation

Production-shaped candidate generation project for commerce retrieval with dense+sparse signals, ANN indexing, and candidate-set quality analysis.

## What This Shows

- Retrieval-focused recommendation systems with shared user/item embeddings
- Candidate generation under recall, coverage, and latency constraints
- Two-tower modeling with public-safe commerce framing
- Candidate handoff quality that can be discussed alongside downstream ranking

## Problem

Commerce recommendation systems rarely score the full catalog directly. They first retrieve a compact candidate set that is broad enough to preserve relevant items and small enough for downstream rankers to handle. This project focuses on that retrieval stage and how to evaluate it honestly.

## Dataset And Candidate Construction

- Primary dataset: Criteo Sponsored Search Conversion Logs
- Fallback dataset: MovieLens 20M with synthetic product and query metadata
- Shared schema: [../../data/README.md](../../data/README.md)

Positive interactions are derived from `click == 1` rows. Training pairs are built by sampling negatives from the item catalog for each positive interaction.

## Models And Baselines

- Popularity baseline
- Matrix factorization baseline
- Two-tower retrieval model

## Public Interfaces

```bash
python src/train.py --config configs/two_tower.yaml
python src/build_index.py --checkpoint checkpoints/two_tower.pt
python src/retrieve.py --user-id 123 --topk 100
python src/evaluate.py --topk 50,100,500
```

## Key Results

This project produces comparison tables for:

- Recall@50 / Recall@100
- MRR@100
- Catalog coverage
- Novelty
- Retrieval latency
- Cold-start slice quality

## Architecture

```text
interaction table
  -> positive interaction extraction
  -> negative sampling
  -> user tower / item tower training
  -> item embedding index build
  -> top-k retrieval
  -> candidate-set evaluation
```

## Trade-offs

- Better recall vs tighter candidate sets
- Richer user and item features vs embedding simplicity
- ANN speed vs retrieval exactness
- Head-item quality vs catalog coverage

## Failure Modes

- Popularity collapse toward head items
- Weak cold-start performance for new users or items
- Retrieval metrics that look good but fail downstream ranking needs
- Index drift when embeddings or item metadata change

## What I Would Improve In Production

- Add hard-negative mining and more realistic sequence-aware user state
- Add feature freshness guarantees and online index update paths
- Connect candidate retrieval metrics to downstream ranker deltas more explicitly
- Benchmark FAISS and ScaNN variants on larger public catalogs

## Testing

```bash
pytest tests -q
```
