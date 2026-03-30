from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_processed_frame
from src.modeling import checkpoint_path, load_bundle, score_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate manual-review queue load.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--events", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frame = load_processed_frame(config, path=args.events)
    eval_frame = frame[frame[config.data["split_column"]] != "train"].reset_index(drop=True)
    bundle = load_bundle(args.model or checkpoint_path(config))
    probabilities, decisions, _ = score_frame(bundle, eval_frame)
    reviewed = eval_frame.assign(score=probabilities, decision=decisions)
    reviewed = reviewed[reviewed["decision"] == "review"].sort_values("score", ascending=False)

    analyst_capacity = int(config.review_queue["analyst_capacity"])
    within_capacity = reviewed.head(analyst_capacity)
    overflow = max(len(reviewed) - analyst_capacity, 0)
    summary = {
        "review_events": int(len(reviewed)),
        "review_rate": float(len(reviewed) / max(len(eval_frame), 1)),
        "within_capacity": int(len(within_capacity)),
        "overflow": int(overflow),
        "fraud_in_review_queue": int(reviewed["is_fraud"].sum()),
        "fraud_within_capacity": int(within_capacity["is_fraud"].sum()),
        "max_review_score": float(reviewed["score"].max()) if not reviewed.empty else 0.0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
