from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_processed_frame
from src.metrics import slice_metrics, summarize_risk_metrics
from src.modeling import checkpoint_path, load_bundle, score_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate payment risk model quality.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frame = load_processed_frame(config)
    test_frame = frame[frame[config.data["split_column"]] == "test"].reset_index(drop=True)
    bundle = load_bundle(args.model or checkpoint_path(config))
    probabilities, decisions, fallback_mask = score_frame(bundle, test_frame)
    payload = {
        "metrics": summarize_risk_metrics(test_frame, probabilities, decisions, config.costs),
        "slice_metrics": slice_metrics(test_frame, probabilities, decisions),
        "fallback_events": int(fallback_mask.sum()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
