# ONNX Low-Latency Serving Lab

Production-shaped serving project for exporting ranking models to ONNX, benchmarking CPU inference, and simulating online feature lookup with cache and timeout behavior.

## What This Shows

- Exporting PyTorch ranking models to ONNX
- Comparing native framework and ONNX Runtime latency
- Mocking online feature access patterns instead of pretending model inference is the whole serving story
- Framing serving quality around latency, parity, memory, and graceful degradation

## Problem

Good offline relevance is not enough if the serving path misses latency targets or depends on brittle feature access. This project treats serving as a systems problem: export compatibility, feature lookup overhead, cache behavior, timeout handling, and benchmarkable CPU inference.

## Model Source

- Primary source: best checkpoint from `../../ranking-systems/01-dcnv2-commerce-ranking/`
- Optional follow-up: export the candidate-generation user tower for retrieval benchmarks

## Public Interfaces

```bash
python src/export.py --checkpoint ../../ranking-systems/01-dcnv2-commerce-ranking/checkpoints/dcnv2.pt
python src/benchmark.py --engine pytorch,onnx --batch-sizes 1,8,32
python src/serve.py --model artifacts/dcnv2.onnx
```

## Key Results

This project produces benchmark tables for:

- p50 / p95 / p99 latency
- Throughput
- Model parity between PyTorch and ONNX
- Memory footprint
- Feature lookup overhead
- Cache sensitivity

## Architecture

```text
ranking checkpoint
  -> ONNX export
  -> runtime selection (PyTorch or ONNX)
  -> mock feature store lookup
  -> cache / timeout handling
  -> score response
  -> latency benchmark summary
```

## Trade-offs

- Better model quality vs stricter latency budgets
- Feature freshness vs lookup overhead
- Cache hit rate vs stale features
- ONNX portability vs export/runtime complexity

## Failure Modes

- Export mismatch between training and serving graphs
- Missing or stale online features
- Cache misses causing p99 regressions
- Runtime dependencies absent in the deployment environment

## What I Would Improve In Production

- Add real rollout hooks and canary/rollback instrumentation
- Benchmark larger batch windows and async request fan-out
- Measure feature-store tail latency separately from model latency
- Add observability for parity drift between training and serving

## Testing

```bash
pytest tests -q
```
