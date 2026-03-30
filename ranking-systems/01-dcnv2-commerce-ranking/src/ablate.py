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

from src.data import load_config, prepare_splits
from src.metrics import summarize_ranking_metrics
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(encoded, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(encoded.categorical, dtype=torch.long),
        torch.as_tensor(encoded.dense, dtype=torch.float32),
        torch.as_tensor(encoded.target, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate feature groups for ranking models.")
    parser.add_argument("--config", default="configs/dcnv2.yaml")
    parser.add_argument("--feature-group", required=True, choices=["sparse", "dense", "crosses"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.training["seed"]))
    encoder, train_encoded, _, test_encoded = prepare_splits(config, feature_group_drop=args.feature_group)
    model = build_model(
        name=config.model["name"],
        cardinalities=encoder.cardinalities,
        dense_dim=len(encoder.dense_features),
        config=config.model,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training["learning_rate"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = build_loader(train_encoded, int(config.training["batch_size"]))

    model.train()
    for _ in range(int(config.training["epochs"])):
        for categorical, dense, labels in loader:
            optimizer.zero_grad()
            logits = model(categorical, dense)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

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
    summary["ablated_group"] = args.feature_group
    summary["remaining_categorical_features"] = encoder.categorical_features
    summary["remaining_dense_features"] = encoder.dense_features
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
