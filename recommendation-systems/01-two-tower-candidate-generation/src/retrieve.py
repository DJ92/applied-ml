from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RetrievalEncoder
from src.index import NumpyANNIndex
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k candidates for a user.")
    parser.add_argument("--checkpoint", default="checkpoints/two_tower.pt")
    parser.add_argument("--index", default="artifacts/two_tower_index.pkl")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder = RetrievalEncoder.from_dict(checkpoint["encoder"])
    user_profiles = checkpoint["user_profiles"]
    profile = user_profiles.get(str(args.user_id))
    if profile is None:
        raise KeyError(f"Unknown user id: {args.user_id}")

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

    user_frame = pd.DataFrame([{**profile}])
    user_cat, user_dense = encoder.encode_user_frame(user_frame)
    with torch.no_grad():
        query_vector = model.user_vector(
            torch.as_tensor(user_cat, dtype=torch.long),
            torch.as_tensor(user_dense, dtype=torch.float32),
        ).cpu().numpy()
    index = NumpyANNIndex.load(args.index)
    items, scores = index.search(query_vector, args.topk)
    print(json.dumps({"user_id": args.user_id, "items": items[0].tolist(), "scores": scores[0].tolist()}, indent=2))


if __name__ == "__main__":
    main()
