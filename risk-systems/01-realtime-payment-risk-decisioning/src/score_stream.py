from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_processed_frame
from src.metrics import latency_summary
from src.modeling import checkpoint_path, load_bundle, score_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a stream of payment events.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--events", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--force-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_bundle(args.model or checkpoint_path(config))
    events = load_processed_frame(config, path=args.events)
    if args.max_events:
        events = events.head(args.max_events).reset_index(drop=True)

    latencies_ms: list[float] = []
    probabilities = []
    decisions = []
    fallback_count = 0
    for idx in range(len(events)):
        row = events.iloc[[idx]]
        start = time.perf_counter()
        row_probs, row_decisions, fallback_mask = score_frame(bundle, row, force_fallback=args.force_fallback)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        probabilities.append(float(row_probs[0]))
        decisions.append(row_decisions[0])
        fallback_count += int(fallback_mask[0])

    summary = {
        "events": int(len(events)),
        "latency_ms": latency_summary(latencies_ms),
        "fallback_events": fallback_count,
        "decision_counts": {
            "approve": decisions.count("approve"),
            "review": decisions.count("review"),
            "decline": decisions.count("decline"),
        },
        "average_score": float(sum(probabilities) / max(len(probabilities), 1)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
