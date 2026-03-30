from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    RetrievalEncoder,
    build_item_catalog,
    build_user_profiles,
    load_config,
    load_frame,
    sample_negatives,
    split_holdout,
    validate_frame,
)
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(encoded, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(encoded.user_categorical, dtype=torch.long),
        torch.as_tensor(encoded.user_dense, dtype=torch.float32),
        torch.as_tensor(encoded.item_categorical, dtype=torch.long),
        torch.as_tensor(encoded.item_dense, dtype=torch.float32),
        torch.as_tensor(encoded.labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-tower candidate generator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config.model["name"] = args.model
    set_seed(int(config.training["seed"]))

    frame = load_frame(config.data["path"])
    validate_frame(frame, config)
    train_df, eval_df = split_holdout(frame, config)
    sampled = sample_negatives(train_df, config, seed=int(config.training["seed"]))
    encoder = RetrievalEncoder(
        config.data["user_categorical_features"],
        config.data["item_categorical_features"],
        config.data["user_dense_features"],
        config.data["item_dense_features"],
    ).fit(pd_concat(train_df, sampled))
    encoded = encoder.encode_pairs(sampled, config.data["target"])
    model = build_model(
        name=config.model["name"],
        config=config.model,
        user_cardinalities=encoder.user_cardinalities,
        item_cardinalities=encoder.item_cardinalities,
        user_dense_dim=len(encoder.user_dense_features),
        item_dense_dim=len(encoder.item_dense_features),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training["learning_rate"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = build_loader(encoded, int(config.training["batch_size"]))

    history = []
    for epoch in range(int(config.training["epochs"])):
        running_loss = 0.0
        model.train()
        for user_cat, user_dense, item_cat, item_dense, labels in loader:
            optimizer.zero_grad()
            logits = model(user_cat, user_dense, item_cat, item_dense)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
        history.append({"epoch": epoch + 1, "train_loss": running_loss / max(len(encoded.labels), 1)})

    item_catalog = build_item_catalog(train_df, config)
    user_profiles = build_user_profiles(train_df, config)
    checkpoint_dir = Path(config.training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.model['name']}.pt"
    torch.save(
        {
            "model_name": config.model["name"],
            "state_dict": model.state_dict(),
            "model_config": config.model,
            "data_config": config.data,
            "training_config": config.training,
            "encoder": encoder.to_dict(),
            "user_cardinalities": encoder.user_cardinalities,
            "item_cardinalities": encoder.item_cardinalities,
            "item_catalog": item_catalog.to_dict(orient="records"),
            "user_profiles": user_profiles,
            "item_popularity": train_df[train_df[config.data["target"]] > 0][config.data["item_id_column"]]
            .astype(str)
            .value_counts()
            .to_dict(),
            "eval_rows": eval_df.to_dict(orient="records"),
        },
        checkpoint_path,
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "history": history}, indent=2))


def pd_concat(frame_a, frame_b):
    import pandas as pd

    return pd.concat([frame_a, frame_b], ignore_index=True)


if __name__ == "__main__":
    main()
