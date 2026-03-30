# Applied ML Systems

> Production-shaped machine learning projects focused on ranking, candidate generation, and serving.

This repository is evolving from a broad applied-ML showcase into a tighter portfolio for staff AI/ML systems roles. The first wave centers on commerce-flavored recommendation and relevance systems that mirror the kinds of problems described in my resume: candidate generation, deep ranking, low-latency serving, and measurable trade-offs.

## Flagship Projects

### 1. DCN/DCN-V2 Commerce Ranking Lab

Public proof for deep ranking, sparse feature handling, embeddings, ablations, calibration, and evaluation rigor.

[-> View Project](ranking-systems/01-dcnv2-commerce-ranking/)

### 2. Two-Tower Candidate Generation

Public proof for embedding-based retrieval, dense+sparse user-item signals, ANN indexing, and candidate handoff quality.

[-> View Project](recommendation-systems/01-two-tower-candidate-generation/)

### 3. ONNX Low-Latency Serving Lab

Public proof for exporting ranking models to ONNX, benchmarking inference paths, and simulating online feature lookup and fallback behavior.

[-> View Project](serving-systems/01-onnx-latency-lab/)

## Shared Data Contract

All three projects share a common interaction-table convention documented in [data/README.md](data/README.md). The primary target dataset is the Criteo Sponsored Search Conversion Logs dataset, with a MovieLens-plus-synthetic-metadata fallback documented in the same place.

## Repository Structure

```text
Applied-ML/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── ranking-systems/
│   └── 01-dcnv2-commerce-ranking/
├── recommendation-systems/
│   └── 01-two-tower-candidate-generation/
└── serving-systems/
    └── 01-onnx-latency-lab/
```

## What This Repo Demonstrates

- Deep ranking systems with realistic sparse and dense feature mixes
- Candidate generation systems that connect retrieval quality to downstream rankers
- Serving-path design under latency budgets, online feature access, and graceful degradation
- Documentation that makes trade-offs, failure modes, and production follow-ups explicit

## Planned Next Areas

Future waves can extend this repo into fraud/risk decisioning, experimentation systems, and classical ML baselines that support system design discussions. For now, the priority is to make the ranking, retrieval, and serving stack concrete and public-safe.
