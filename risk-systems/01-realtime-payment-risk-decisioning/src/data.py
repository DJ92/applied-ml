from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class RiskConfig:
    data: dict[str, Any]
    features: dict[str, list[str]]
    thresholds: dict[str, float]
    review_queue: dict[str, Any]
    costs: dict[str, float]
    training: dict[str, Any]


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "risk_model.yaml"
RAW_REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


def load_config(path: str | Path | None = None) -> RiskConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return RiskConfig(
        data=raw["data"],
        features=raw["features"],
        thresholds=raw["thresholds"],
        review_queue=raw["review_queue"],
        costs=raw["costs"],
        training=raw["training"],
    )


def load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Could not find dataset at {source}.")
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source)


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    return destination


def validate_raw_frame(frame: pd.DataFrame) -> None:
    missing = sorted(RAW_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")


def normalize_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_raw_frame(frame)
    normalized = frame.rename(
        columns={
            "nameOrig": "customer_id",
            "nameDest": "destination_id",
            "oldbalanceOrg": "oldbalance_orig",
            "newbalanceOrig": "newbalance_orig",
            "oldbalanceDest": "oldbalance_dest",
            "newbalanceDest": "newbalance_dest",
            "isFraud": "is_fraud",
            "isFlaggedFraud": "is_flagged_fraud",
        }
    ).copy()
    normalized["type"] = normalized["type"].astype(str)
    normalized["customer_id"] = normalized["customer_id"].astype(str)
    normalized["destination_id"] = normalized["destination_id"].astype(str)
    numeric_columns = [
        "step",
        "amount",
        "oldbalance_orig",
        "newbalance_orig",
        "oldbalance_dest",
        "newbalance_dest",
        "is_fraud",
        "is_flagged_fraud",
    ]
    normalized[numeric_columns] = normalized[numeric_columns].apply(pd.to_numeric)
    normalized = normalized.sort_values(["step", "customer_id", "destination_id"]).reset_index(drop=True)
    return normalized


def _prior_group_features(frame: pd.DataFrame, group_col: str, value_col: str, fraud_col: str) -> pd.DataFrame:
    ordered = frame.sort_values(["step", group_col]).copy()
    counts = ordered.groupby(group_col).cumcount().astype("float32")
    cumulative_value = ordered.groupby(group_col)[value_col].cumsum() - ordered[value_col]
    cumulative_fraud = ordered.groupby(group_col)[fraud_col].cumsum() - ordered[fraud_col]
    previous_step = ordered.groupby(group_col)["step"].shift(1)
    ordered["prior_txn_count"] = counts
    ordered["avg_amount"] = np.divide(
        cumulative_value,
        counts,
        out=np.zeros(len(ordered), dtype=np.float32),
        where=counts.to_numpy() > 0,
    )
    ordered["prior_fraud_rate"] = np.divide(
        cumulative_fraud,
        counts,
        out=np.zeros(len(ordered), dtype=np.float32),
        where=counts.to_numpy() > 0,
    )
    ordered["hours_since_previous"] = (ordered["step"] - previous_step).fillna(999.0).astype("float32")
    return ordered.sort_index()


def derive_features(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_raw_frame(raw_frame)
    frame["hour_of_day"] = (frame["step"] % 24).astype("float32")
    frame["day_index"] = np.floor_divide(frame["step"], 24).astype("float32")
    frame["origin_balance_gap"] = (
        frame["newbalance_orig"] - (frame["oldbalance_orig"] - frame["amount"])
    ).astype("float32")
    frame["destination_balance_gap"] = (
        frame["newbalance_dest"] - (frame["oldbalance_dest"] + frame["amount"])
    ).astype("float32")
    frame["is_flagged_fraud_signal"] = frame["is_flagged_fraud"].astype("float32")

    customer_features = _prior_group_features(frame, "customer_id", "amount", "is_fraud")
    destination_features = _prior_group_features(frame, "destination_id", "amount", "is_fraud")

    frame["customer_prior_txn_count"] = customer_features["prior_txn_count"]
    frame["customer_avg_amount"] = customer_features["avg_amount"].astype("float32")
    frame["customer_prior_fraud_rate"] = customer_features["prior_fraud_rate"].astype("float32")
    frame["hours_since_prev_customer_txn"] = customer_features["hours_since_previous"].astype("float32")
    frame["destination_prior_txn_count"] = destination_features["prior_txn_count"]
    frame["destination_avg_amount"] = destination_features["avg_amount"].astype("float32")
    return frame


def split_by_time(frame: pd.DataFrame, config: RiskConfig) -> pd.DataFrame:
    ratios = config.data["split_ratios"]
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")
    time_col = config.data["time_column"]
    unique_steps = np.sort(frame[time_col].unique())
    if len(unique_steps) < 3:
        raise ValueError("Need at least three distinct timesteps to create train/val/test splits.")
    train_idx = max(1, int(len(unique_steps) * ratios[0])) - 1
    val_idx = max(train_idx + 1, int(len(unique_steps) * (ratios[0] + ratios[1]))) - 1
    train_cut = unique_steps[min(train_idx, len(unique_steps) - 3)]
    val_cut = unique_steps[min(val_idx, len(unique_steps) - 2)]

    output = frame.copy()
    output[config.data["split_column"]] = np.where(
        output[time_col] <= train_cut,
        "train",
        np.where(output[time_col] <= val_cut, "val", "test"),
    )
    return output


def required_processed_columns(config: RiskConfig) -> list[str]:
    return sorted(
        {
            config.data["time_column"],
            config.data["target"],
            config.data["split_column"],
            *config.features["categorical_features"],
            *config.features["dense_features"],
        }
    )


def prepare_processed_frame(
    config: RiskConfig,
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    raw = load_frame(raw_path or config.data["raw_path"])
    featured = derive_features(raw)
    featured = split_by_time(featured, config)
    featured[config.data["target"]] = featured[config.data["target"]].astype("int64")
    destination = write_frame(featured, output_path or config.data["processed_path"])
    return featured, destination


def load_processed_frame(config: RiskConfig, path: str | Path | None = None) -> pd.DataFrame:
    frame = load_frame(path or config.data["processed_path"])
    missing = [column for column in required_processed_columns(config) if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")
    return frame
