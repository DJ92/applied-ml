from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class RankingConfig:
    data: dict[str, Any]
    feature_groups: dict[str, list[str]]
    training: dict[str, Any]
    model: dict[str, Any]


@dataclass
class EncodedFrame:
    categorical: np.ndarray
    dense: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    raw: pd.DataFrame


def load_config(path: str | Path) -> RankingConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return RankingConfig(
        data=raw["data"],
        feature_groups=raw["feature_groups"],
        training=raw["training"],
        model=raw["model"],
    )


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {path}. See ../../data/README.md for the shared schema."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def apply_cross_features(frame: pd.DataFrame, cross_features: list[dict[str, Any]]) -> pd.DataFrame:
    output = frame.copy()
    for feature in cross_features:
        name = feature["name"]
        cols = feature["columns"]
        output[name] = output[cols].astype(str).agg("__x__".join, axis=1)
    return output


def required_columns(config: RankingConfig) -> list[str]:
    cols = [
        *config.data["categorical_features"],
        *config.data["dense_features"],
        config.data["target"],
        config.data["group_column"],
    ]
    for cross in config.data.get("cross_features", []):
        cols.extend(cross["columns"])
    return sorted(set(cols))


def validate_frame(frame: pd.DataFrame, config: RankingConfig) -> None:
    missing = [col for col in required_columns(config) if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def split_by_group(
    frame: pd.DataFrame, group_column: str, split_ratios: list[float], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = frame[group_column].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_cut = int(len(groups) * split_ratios[0])
    val_cut = train_cut + int(len(groups) * split_ratios[1])
    train_groups = set(groups[:train_cut])
    val_groups = set(groups[train_cut:val_cut])
    train = frame[frame[group_column].astype(str).isin(train_groups)].reset_index(drop=True)
    val = frame[frame[group_column].astype(str).isin(val_groups)].reset_index(drop=True)
    test = frame[
        ~frame[group_column].astype(str).isin(train_groups | val_groups)
    ].reset_index(drop=True)
    return train, val, test


class FeatureEncoder:
    def __init__(
        self,
        categorical_features: list[str],
        dense_features: list[str],
        target: str,
        group_column: str,
    ) -> None:
        self.categorical_features = categorical_features
        self.dense_features = dense_features
        self.target = target
        self.group_column = group_column
        self.category_maps: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        for feature in self.categorical_features:
            values = frame[feature].astype(str).fillna("__missing__").unique()
            self.category_maps[feature] = {value: idx + 1 for idx, value in enumerate(sorted(values))}
        return self

    def transform(self, frame: pd.DataFrame) -> EncodedFrame:
        categorical = []
        for feature in self.categorical_features:
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
            categorical.append(encoded)
        cat_matrix = np.stack(categorical, axis=1) if categorical else np.zeros((len(frame), 0), dtype=np.int64)
        dense = frame[self.dense_features].fillna(0.0).astype("float32").to_numpy()
        target = frame[self.target].fillna(0.0).astype("float32").to_numpy()
        groups = frame[self.group_column].astype(str).to_numpy()
        return EncodedFrame(cat_matrix, dense, target, groups, frame.reset_index(drop=True))

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.category_maps[feature]) + 1 for feature in self.categorical_features]

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical_features": self.categorical_features,
            "dense_features": self.dense_features,
            "target": self.target,
            "group_column": self.group_column,
            "category_maps": self.category_maps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureEncoder":
        encoder = cls(
            categorical_features=payload["categorical_features"],
            dense_features=payload["dense_features"],
            target=payload["target"],
            group_column=payload["group_column"],
        )
        encoder.category_maps = payload["category_maps"]
        return encoder


def prepare_splits(config: RankingConfig, feature_group_drop: str | None = None) -> tuple[FeatureEncoder, EncodedFrame, EncodedFrame, EncodedFrame]:
    frame = load_frame(config.data["path"])
    validate_frame(frame, config)
    frame = apply_cross_features(frame, config.data.get("cross_features", []))

    categorical = list(config.data["categorical_features"])
    dense = list(config.data["dense_features"])
    cross_names = [cross["name"] for cross in config.data.get("cross_features", [])]

    if feature_group_drop:
        group_features = set(config.feature_groups.get(feature_group_drop, []))
        categorical = [feature for feature in categorical if feature not in group_features]
        dense = [feature for feature in dense if feature not in group_features]
        cross_names = [feature for feature in cross_names if feature not in group_features]

    train_df, val_df, test_df = split_by_group(
        frame=frame,
        group_column=config.data["group_column"],
        split_ratios=config.data["split_ratios"],
        seed=int(config.training["seed"]),
    )
    encoder = FeatureEncoder(
        categorical_features=[*categorical, *cross_names],
        dense_features=dense,
        target=config.data["target"],
        group_column=config.data["group_column"],
    ).fit(train_df)
    return encoder, encoder.transform(train_df), encoder.transform(val_df), encoder.transform(test_df)
