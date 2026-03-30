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
from src.index import build_best_available_index
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ANN index from a trained retrieval model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/two_tower_index.pkl")
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

    item_catalog = pd.DataFrame(checkpoint["item_catalog"])
    item_cat, item_dense = encoder.encode_item_frame(item_catalog)
    with torch.no_grad():
        item_vectors = model.item_vector(
            torch.as_tensor(item_cat, dtype=torch.long),
            torch.as_tensor(item_dense, dtype=torch.float32),
        ).cpu().numpy()
    item_ids = item_catalog[checkpoint["data_config"]["item_id_column"]].astype(str).tolist()
    index = build_best_available_index(item_ids, item_vectors)
    path = index.save(args.output)
    print(json.dumps({"index_path": str(path), "item_count": len(item_ids)}, indent=2))


if __name__ == "__main__":
    main()
