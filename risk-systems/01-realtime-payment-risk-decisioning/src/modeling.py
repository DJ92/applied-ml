from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import RiskConfig


def feature_columns(config: RiskConfig) -> list[str]:
    return [*config.features["categorical_features"], *config.features["dense_features"]]


def split_frame(frame: pd.DataFrame, split_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame[split_column] == "train"].reset_index(drop=True)
    val = frame[frame[split_column] == "val"].reset_index(drop=True)
    test = frame[frame[split_column] == "test"].reset_index(drop=True)
    return train, val, test


def build_pipeline(config: RiskConfig) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config.features["categorical_features"],
            ),
            ("dense", StandardScaler(), config.features["dense_features"]),
        ]
    )
    classifier = LogisticRegression(
        max_iter=int(config.training["max_iter"]),
        class_weight=config.training.get("class_weight"),
        random_state=int(config.training["seed"]),
    )
    return Pipeline([("transform", transformer), ("model", classifier)])


def train_bundle(frame: pd.DataFrame, config: RiskConfig) -> dict[str, Any]:
    train, val, test = split_frame(frame, config.data["split_column"])
    pipeline = build_pipeline(config)
    columns = feature_columns(config)
    target = config.data["target"]
    pipeline.fit(train[columns], train[target])
    return {
        "pipeline": pipeline,
        "thresholds": config.thresholds,
        "costs": config.costs,
        "feature_columns": columns,
        "categorical_features": config.features["categorical_features"],
        "dense_features": config.features["dense_features"],
        "target": target,
        "split_column": config.data["split_column"],
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }


def checkpoint_path(config: RiskConfig, name: str = "risk_model.pkl") -> Path:
    checkpoint_dir = Path(config.training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / name


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        pickle.dump(bundle, handle)
    return destination


def load_bundle(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def heuristic_score(row: dict[str, Any]) -> float:
    risk = 0.02
    risk += min(float(row.get("amount", 0.0)) / 20000.0, 0.25)
    risk += 0.5 * float(row.get("is_flagged_fraud_signal", 0.0))
    risk += 0.15 if float(row.get("origin_balance_gap", 0.0)) < -1.0 else 0.0
    risk += min(float(row.get("customer_prior_fraud_rate", 0.0)) * 0.5, 0.2)
    return float(np.clip(risk, 0.0, 0.99))


def decision_from_score(score: float, thresholds: dict[str, float]) -> str:
    if score >= float(thresholds["review"]):
        return "decline"
    if score >= float(thresholds["approve"]):
        return "review"
    return "approve"


def score_frame(bundle: dict[str, Any], frame: pd.DataFrame, force_fallback: bool = False) -> tuple[np.ndarray, list[str], np.ndarray]:
    pipeline = bundle.get("pipeline")
    columns = bundle["feature_columns"]
    fallback_mask = frame[columns].isna().any(axis=1).to_numpy() | force_fallback
    probabilities = np.zeros(len(frame), dtype=np.float32)

    if pipeline is not None and (~fallback_mask).any():
        probabilities[~fallback_mask] = pipeline.predict_proba(frame.loc[~fallback_mask, columns])[:, 1]

    for idx, use_fallback in enumerate(fallback_mask):
        if use_fallback:
            probabilities[idx] = heuristic_score(frame.iloc[idx].to_dict())

    decisions = [decision_from_score(float(score), bundle["thresholds"]) for score in probabilities]
    return probabilities, decisions, fallback_mask.astype(bool)
