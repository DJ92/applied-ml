from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_processed_frame
from src.metrics import summarize_risk_metrics
from src.modeling import checkpoint_path, save_bundle, score_frame, split_frame, train_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a payment risk decisioning model.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.training["seed"]))
    frame = load_processed_frame(config)
    bundle = train_bundle(frame, config)
    path = save_bundle(bundle, checkpoint_path(config))
    _, val_frame, test_frame = split_frame(frame, config.data["split_column"])
    val_probs, val_decisions, _ = score_frame(bundle, val_frame)
    test_probs, test_decisions, _ = score_frame(bundle, test_frame)
    payload = {
        "checkpoint": str(path),
        "validation": summarize_risk_metrics(val_frame, val_probs, val_decisions, config.costs),
        "test": summarize_risk_metrics(test_frame, test_probs, test_decisions, config.costs),
        "splits": bundle["splits"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
