from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class RetrievalConfig:
    data: dict[str, Any]
    training: dict[str, Any]
    model: dict[str, Any]


@dataclass
class EncodedExamples:
    user_categorical: np.ndarray
    user_dense: np.ndarray
    item_categorical: np.ndarray
    item_dense: np.ndarray
    labels: np.ndarray


def load_config(path: str | Path) -> RetrievalConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return RetrievalConfig(data=raw["data"], training=raw["training"], model=raw["model"])


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}. See ../../data/README.md.")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def required_columns(config: RetrievalConfig) -> list[str]:
    return sorted(
        set(
            [
                config.data["user_id_column"],
                config.data["item_id_column"],
                config.data["target"],
                *config.data["user_categorical_features"],
                *config.data["item_categorical_features"],
                *config.data["user_dense_features"],
                *config.data["item_dense_features"],
            ]
        )
    )


def validate_frame(frame: pd.DataFrame, config: RetrievalConfig) -> None:
    missing = [col for col in required_columns(config) if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def split_holdout(frame: pd.DataFrame, config: RetrievalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_col = config.data["user_id_column"]
    target_col = config.data["target"]
    ordered = frame.reset_index(drop=True).copy()
    if "event_ts" in ordered.columns:
        ordered = ordered.sort_values("event_ts")
    positives = ordered[ordered[target_col] > 0].copy()
    eval_indices = []
    for _, user_rows in positives.groupby(user_col):
        if len(user_rows) < 2:
            continue
        eval_indices.append(user_rows.index[-1])
    eval_df = ordered.loc[sorted(eval_indices)].reset_index(drop=True)
    train_df = ordered.drop(index=eval_indices).reset_index(drop=True)
    return train_df, eval_df


def build_user_profiles(train_df: pd.DataFrame, config: RetrievalConfig) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    user_col = config.data["user_id_column"]
    for user_id, group in train_df.groupby(user_col):
        profile = {}
        for feature in config.data["user_categorical_features"]:
            profile[feature] = str(group[feature].mode(dropna=False).iloc[0])
        for feature in config.data["user_dense_features"]:
            profile[feature] = float(group[feature].fillna(0.0).mean())
        profiles[str(user_id)] = profile
    return profiles


def build_item_catalog(frame: pd.DataFrame, config: RetrievalConfig) -> pd.DataFrame:
    item_col = config.data["item_id_column"]
    catalog = frame.sort_values(item_col).drop_duplicates(subset=[item_col], keep="last")
    return catalog.reset_index(drop=True)


def sample_negatives(train_df: pd.DataFrame, config: RetrievalConfig, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_col = config.data["user_id_column"]
    item_col = config.data["item_id_column"]
    target_col = config.data["target"]
    all_items = train_df[item_col].astype(str).unique()
    positives = train_df[train_df[target_col] > 0].reset_index(drop=True)
    rows = []
    history = train_df.groupby(user_col)[item_col].apply(lambda values: set(values.astype(str))).to_dict()
    for _, row in positives.iterrows():
        rows.append({**row.to_dict(), target_col: 1})
        available = [item for item in all_items if item not in history[row[user_col]]]
        sample_size = min(len(available), int(config.data["negative_samples"]))
        for negative_item in rng.choice(available, size=sample_size, replace=False):
            cloned = row.to_dict()
            cloned[item_col] = negative_item
            item_template = train_df[train_df[item_col].astype(str) == str(negative_item)].iloc[0]
            for feature in config.data["item_categorical_features"] + config.data["item_dense_features"]:
                cloned[feature] = item_template[feature]
            cloned[target_col] = 0
            rows.append(cloned)
    return pd.DataFrame(rows)


class RetrievalEncoder:
    def __init__(
        self,
        user_categorical_features: list[str],
        item_categorical_features: list[str],
        user_dense_features: list[str],
        item_dense_features: list[str],
    ) -> None:
        self.user_categorical_features = user_categorical_features
        self.item_categorical_features = item_categorical_features
        self.user_dense_features = user_dense_features
        self.item_dense_features = item_dense_features
        self.category_maps: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "RetrievalEncoder":
        for feature in self.user_categorical_features + self.item_categorical_features:
            values = frame[feature].astype(str).fillna("__missing__").unique()
            self.category_maps[feature] = {value: idx + 1 for idx, value in enumerate(sorted(values))}
        return self

    def _encode_columns(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        values = []
        for feature in features:
            mapping = self.category_maps[feature]
            encoded = (
                frame[feature]
                .astype(str)
                .fillna("__missing__")
                .map(mapping)
                .fillna(0)
                .astype("int64")
                .to_numpy()
            )
            values.append(encoded)
        return np.stack(values, axis=1) if values else np.zeros((len(frame), 0), dtype=np.int64)

    def encode_pairs(self, frame: pd.DataFrame, target: str) -> EncodedExamples:
        return EncodedExamples(
            user_categorical=self._encode_columns(frame, self.user_categorical_features),
            user_dense=frame[self.user_dense_features].fillna(0.0).astype("float32").to_numpy(),
            item_categorical=self._encode_columns(frame, self.item_categorical_features),
            item_dense=frame[self.item_dense_features].fillna(0.0).astype("float32").to_numpy(),
            labels=frame[target].fillna(0.0).astype("float32").to_numpy(),
        )

    def encode_user_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._encode_columns(frame, self.user_categorical_features),
            frame[self.user_dense_features].fillna(0.0).astype("float32").to_numpy(),
        )

    def encode_item_frame(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._encode_columns(frame, self.item_categorical_features),
            frame[self.item_dense_features].fillna(0.0).astype("float32").to_numpy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_categorical_features": self.user_categorical_features,
            "item_categorical_features": self.item_categorical_features,
            "user_dense_features": self.user_dense_features,
            "item_dense_features": self.item_dense_features,
            "category_maps": self.category_maps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalEncoder":
        encoder = cls(
            payload["user_categorical_features"],
            payload["item_categorical_features"],
            payload["user_dense_features"],
            payload["item_dense_features"],
        )
        encoder.category_maps = payload["category_maps"]
        return encoder

    @property
    def user_cardinalities(self) -> list[int]:
        return [len(self.category_maps[feature]) + 1 for feature in self.user_categorical_features]

    @property
    def item_cardinalities(self) -> list[int]:
        return [len(self.category_maps[feature]) + 1 for feature in self.item_categorical_features]
