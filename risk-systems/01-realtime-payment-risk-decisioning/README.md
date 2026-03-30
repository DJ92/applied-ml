# Real-Time Payment Risk Decisioning

Production-shaped fraud and risk project for payment flows, with event-time feature engineering, thresholded decisions, manual-review simulation, and latency-aware stream scoring.

## What This Shows

- Real-time risk scoring for payment events instead of offline-only fraud classification
- Balancing fraud capture against customer friction and review load
- Event-time features, fallback rules, and operational decision thresholds
- Framing model quality around latency, false positives, and review-queue pressure

## Problem

Fraud and risk systems are not just classifiers. They sit in the middle of real payment flows where latency, false positives, and degraded feature paths matter as much as raw predictive power. This project uses a PaySim-style dataset to approximate that environment with approve, review, and decline decisions.

## Dataset And Feature Schema

- Primary dataset: PaySim
- Shared repository contract: [../../data/README.md](../../data/README.md)
- Raw inputs mirror PaySim transaction columns such as `step`, `type`, balances, and fraud labels
- Derived features include:
  - event-time recency and day/hour context
  - customer and destination prior transaction counts
  - historical amount summaries
  - prior fraud-rate features
  - balance-gap anomaly signals

## Public Interfaces

```bash
python src/prepare_data.py
python src/train.py --config configs/risk_model.yaml
python src/score_stream.py --events ../../data/processed/payment_risk_events.csv
python src/evaluate.py
python src/simulate_review_queue.py
```

## Key Results

This project produces summary tables for:

- AUC
- precision / recall for flagged transactions
- fraud capture and false-positive rate
- review rate and decline rate
- average expected cost per transaction
- stream scoring p50 / p95 / p99 latency

## Architecture

```text
raw payment events
  -> event-time feature engineering
  -> deterministic train/val/test split by step
  -> logistic risk model + heuristic fallback rules
  -> approve / review / decline thresholds
  -> manual-review queue simulation
  -> latency and slice-metric reporting
```

## Trade-offs

- Higher fraud capture vs more customer friction
- Richer online features vs simpler operational dependencies
- More review traffic vs fewer false declines
- Better model quality vs stricter scoring latency

## Failure Modes

- Temporal leakage from non-causal feature construction
- Review queue overload during fraud spikes
- Fraud drift that invalidates static thresholds
- Missing online features forcing fallback behavior

## What I Would Improve In Production

- Add a gradient-boosted baseline and compare against the linear model
- Add feature freshness monitoring and degraded-mode alerts
- Model analyst feedback loops and threshold retuning over time
- Add event-driven serving hooks and canary rollout instrumentation

## Testing

```bash
pytest tests -q
```
