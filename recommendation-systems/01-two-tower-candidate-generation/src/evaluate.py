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

from src.data import RetrievalEncoder
from src.index import NumpyANNIndex
from src.models import build_model


def recall_at_k(recommended: list[str], truth: str, k: int) -> float:
    return 1.0 if truth in recommended[:k] else 0.0


def mrr_at_k(recommended: list[str], truth: str, k: int) -> float:
    for index, item in enumerate(recommended[:k], start=1):
        if item == truth:
            return 1.0 / index
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a candidate generator.")
    parser.add_argument("--checkpoint", default="checkpoints/two_tower.pt")
    parser.add_argument("--index", default="artifacts/two_tower_index.pkl")
    parser.add_argument("--topk", default="50,100,500")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder = RetrievalEncoder.from_dict(checkpoint["encoder"])
    model = build_model(
        checkpoint["model_name"],
        checkpoint["model_config"],
        checkpoint["user_cardinalities"],
        checkpoint["item_cardinalities"],
        len(encoder.user_dense_features),
        len(encoder.item_dense_features),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    index = NumpyANNIndex.load(args.index)

    eval_rows = pd.DataFrame(checkpoint["eval_rows"])
    if eval_rows.empty:
        raise ValueError("No evaluation rows were saved in the checkpoint.")
    user_cat, user_dense = encoder.encode_user_frame(eval_rows)
    with torch.no_grad():
        user_vectors = model.user_vector(
            torch.as_tensor(user_cat, dtype=torch.long),
            torch.as_tensor(user_dense, dtype=torch.float32),
        ).cpu().numpy()
    max_topk = max(int(value) for value in args.topk.split(","))
    items, _ = index.search(user_vectors, max_topk)

    item_col = checkpoint["data_config"]["item_id_column"]
    popularity = checkpoint["item_popularity"]
    topks = [int(value) for value in args.topk.split(",")]
    metrics: dict[str, float] = {}
    for k in topks:
        recalls = [recall_at_k(row.tolist(), truth, k) for row, truth in zip(items, eval_rows[item_col].astype(str))]
        mrrs = [mrr_at_k(row.tolist(), truth, k) for row, truth in zip(items, eval_rows[item_col].astype(str))]
        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"mrr@{k}"] = float(np.mean(mrrs))

    unique_items = set(items[:, : topks[-1]].reshape(-1).tolist())
    metrics["catalog_coverage"] = float(len(unique_items) / max(len(index.item_ids), 1))
    novelty_scores = []
    for row in items[:, : topks[-1]]:
        for item in row.tolist():
            novelty_scores.append(1.0 / (1.0 + popularity.get(item, 0)))
    metrics["novelty"] = float(np.mean(novelty_scores))
    user_counts = eval_rows[checkpoint["data_config"]["user_id_column"]].astype(str).value_counts()
    metrics["cold_start_user_share"] = float((user_counts == 1).mean())
    metrics["candidate_set_hit_rate_vs_full_catalog_ceiling"] = metrics.get(f"recall@{topks[-1]}", 0.0)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
