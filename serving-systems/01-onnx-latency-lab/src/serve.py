from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_store import MockFeatureStore
from src.runtime import (
    load_metadata,
    load_ranking_bundle,
    load_ranking_model,
    sample_row_from_encoder,
    vectorize_row,
)


def build_request(encoder_payload: dict, user_id: str, product_id: str) -> dict:
    request = sample_row_from_encoder(encoder_payload)
    request["user_id"] = user_id
    request["product_id"] = product_id
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal scoring path with mocked feature lookup.")
    parser.add_argument("--model", default="artifacts/dcnv2.onnx")
    parser.add_argument("--checkpoint", default="../../ranking-systems/01-dcnv2-commerce-ranking/checkpoints/dcnv2.pt")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--product-id", default="demo-product")
    parser.add_argument("--request-file", default=None)
    args = parser.parse_args()

    if args.request_file:
        request = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
    elif args.model.endswith(".onnx"):
        metadata = load_metadata(args.model)
        request = build_request(metadata["encoder"], args.user_id, args.product_id)
    else:
        checkpoint = load_ranking_bundle(args.checkpoint)
        request = build_request(checkpoint["encoder"], args.user_id, args.product_id)

    user_store = MockFeatureStore(
        records={f"user:{request['user_id']}": {"user_id": request["user_id"], "user_recency_days": request.get("user_recency_days", 0.0)}}
    )
    item_store = MockFeatureStore(
        records={
            f"item:{request['product_id']}": {
                "product_id": request["product_id"],
                "category_id": request.get("category_id", "unknown-category"),
                "price": request.get("price", 0.0),
                "discount": request.get("discount", 0.0),
                "product_age_days": request.get("product_age_days", 0.0),
            }
        }
    )
    user_features = user_store.get(f"user:{request['user_id']}")
    item_features = item_store.get(f"item:{request['product_id']}")
    merged = {**request, **user_features, **item_features}

    if args.model.endswith(".onnx"):
        import onnxruntime as ort

        metadata = load_metadata(args.model)
        categorical, dense = vectorize_row(merged, metadata["encoder"])
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        logits = session.run(
            None,
            {
                "categorical_features": categorical.astype("int64"),
                "dense_features": dense.astype("float32"),
            },
        )[0]
        score = float(1.0 / (1.0 + np.exp(-logits[0])))
    else:
        bundle = load_ranking_bundle(args.checkpoint)
        model = load_ranking_model(bundle)
        categorical, dense = vectorize_row(merged, bundle["encoder"])
        with torch.no_grad():
            logits = model(
                torch.as_tensor(categorical, dtype=torch.long),
                torch.as_tensor(dense, dtype=torch.float32),
            ).cpu().numpy()
        score = float(1.0 / (1.0 + np.exp(-logits[0])))

    print(
        json.dumps(
            {
                "user_id": request["user_id"],
                "product_id": request["product_id"],
                "score": score,
                "feature_store_stats": {
                    "user": user_store.stats(),
                    "item": item_store.stats(),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
