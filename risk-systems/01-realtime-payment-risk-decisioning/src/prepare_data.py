from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, prepare_processed_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PaySim-style payment risk data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frame, destination = prepare_processed_frame(config, raw_path=args.input, output_path=args.output)
    summary = {
        "output_path": str(destination),
        "rows": int(len(frame)),
        "fraud_rate": float(frame[config.data["target"]].mean()),
        "split_counts": frame[config.data["split_column"]].value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
