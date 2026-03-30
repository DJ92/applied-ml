# Applied ML Systems

> Production-shaped machine learning projects focused on ranking, candidate generation, serving, and risk decisioning.

This repository is evolving from a broad applied-ML showcase into a tighter portfolio for staff AI/ML systems roles. The first waves center on production-shaped recommender and payments problems that mirror the kinds of systems described in my resume: candidate generation, deep ranking, low-latency serving, and real-time risk decisioning.

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

### 4. Real-Time Payment Risk Decisioning

Public proof for fraud capture vs customer friction trade-offs, event-time feature engineering, review-queue management, and low-latency scoring paths in payment flows.

[-> View Project](risk-systems/01-realtime-payment-risk-decisioning/)

## Shared Data Contract

The recommender projects share a common interaction-table convention documented in [data/README.md](data/README.md). The risk project adds a second public-safe dataset path for PaySim-style payment events so the repository now covers both recommendation and fraud/risk systems.

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
├── risk-systems/
│   └── 01-realtime-payment-risk-decisioning/
└── serving-systems/
    └── 01-onnx-latency-lab/
```

## What This Repo Demonstrates

- Deep ranking systems with realistic sparse and dense feature mixes
- Candidate generation systems that connect retrieval quality to downstream rankers
- Serving-path design under latency budgets, online feature access, and graceful degradation
- Fraud and risk decisioning systems that expose latency, thresholding, and manual-review trade-offs
- Documentation that makes trade-offs, failure modes, and production follow-ups explicit

## Planned Next Areas

Future waves can extend this repo into experimentation systems, classical ML baselines, and broader decisioning simulations that support system design discussions. For now, the priority is to make the ranking, retrieval, serving, and risk stack concrete and public-safe.
