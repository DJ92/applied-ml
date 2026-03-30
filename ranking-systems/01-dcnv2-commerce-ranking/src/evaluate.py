from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FeatureEncoder, EncodedFrame, load_config, prepare_splits
from src.metrics import summarize_ranking_metrics
from src.models import build_model


def slice_metrics(frame: EncodedFrame, logits: np.ndarray) -> dict[str, dict[str, float]]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    output: dict[str, dict[str, float]] = {}
    if "device_type" in frame.raw.columns:
        output["device_type"] = {
            str(device): float(frame.raw.loc[frame.raw["device_type"] == device, "click"].mean())
            for device in sorted(frame.raw["device_type"].astype(str).unique())
        }
    if "category_id" in frame.raw.columns:
        top_categories = frame.raw["category_id"].astype(str).value_counts().head(3).index
        output["category_ctr"] = {
            category: float(frame.raw.loc[frame.raw["category_id"].astype(str) == category, "click"].mean())
            for category in top_categories
        }
    if "user_id" in frame.raw.columns:
        counts = frame.raw["user_id"].astype(str).value_counts()
        freq = frame.raw["user_id"].astype(str).map(counts)
        bucket = pd.cut(freq, bins=[0, 2, 5, np.inf], labels=["low", "mid", "high"])
        output["user_frequency_bucket"] = {
            str(label): float(probs[bucket == label].mean()) for label in bucket.cat.categories if np.any(bucket == label)
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ranking model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/dcnv2.yaml")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    config = load_config(args.config)
    encoder, _, _, test_encoded = prepare_splits(config)
    encoder = FeatureEncoder.from_dict(checkpoint["encoder"])
    model = build_model(
        name=checkpoint["model_name"],
        cardinalities=checkpoint["cardinalities"],
        dense_dim=checkpoint["dense_dim"],
        config=checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.as_tensor(test_encoded.categorical, dtype=torch.long),
            torch.as_tensor(test_encoded.dense, dtype=torch.float32),
        ).cpu().numpy()
    summary = summarize_ranking_metrics(
        test_encoded.target,
        logits,
        test_encoded.groups,
        model=model,
        categorical=test_encoded.categorical,
        dense=test_encoded.dense,
    )
    summary["slices"] = slice_metrics(test_encoded, logits)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
