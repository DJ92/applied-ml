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

from src.data import FeatureEncoder, load_config, prepare_splits
from src.metrics import summarize_ranking_metrics
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loader(encoded, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(encoded.categorical, dtype=torch.long),
        torch.as_tensor(encoded.dense, dtype=torch.float32),
        torch.as_tensor(encoded.target, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict_logits(model: torch.nn.Module, encoded) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        categorical = torch.as_tensor(encoded.categorical, dtype=torch.long)
        dense = torch.as_tensor(encoded.dense, dtype=torch.float32)
        logits = model(categorical, dense).cpu().numpy()
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a commerce ranking model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config.model["name"] = args.model
    set_seed(int(config.training["seed"]))
    encoder, train_encoded, val_encoded, test_encoded = prepare_splits(config)

    model = build_model(
        name=config.model["name"],
        cardinalities=encoder.cardinalities,
        dense_dim=len(encoder.dense_features),
        config=config.model,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_loader = build_loader(train_encoded, batch_size=int(config.training["batch_size"]), shuffle=True)

    history = []
    for epoch in range(int(config.training["epochs"])):
        model.train()
        running_loss = 0.0
        for categorical, dense, labels in train_loader:
            optimizer.zero_grad()
            logits = model(categorical, dense)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
        train_loss = running_loss / max(len(train_encoded.target), 1)
        val_logits = predict_logits(model, val_encoded)
        val_metrics = summarize_ranking_metrics(val_encoded.target, val_logits, val_encoded.groups)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

    checkpoint_dir = Path(config.training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.model['name']}.pt"
    payload = {
        "model_name": config.model["name"],
        "state_dict": model.state_dict(),
        "model_config": config.model,
        "encoder": encoder.to_dict(),
        "cardinalities": encoder.cardinalities,
        "dense_dim": len(encoder.dense_features),
        "training_config": config.training,
        "data_config": config.data,
    }
    torch.save(payload, checkpoint_path)

    test_logits = predict_logits(model, test_encoded)
    test_metrics = summarize_ranking_metrics(
        test_encoded.target,
        test_logits,
        test_encoded.groups,
        model=model,
        categorical=test_encoded.categorical,
        dense=test_encoded.dense,
    )
    summary = {"checkpoint": str(checkpoint_path), "history": history, "test_metrics": test_metrics}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
