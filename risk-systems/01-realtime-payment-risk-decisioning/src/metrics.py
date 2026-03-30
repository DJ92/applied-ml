from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = np.asarray(latencies_ms, dtype=np.float32)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def expected_cost(y_true: np.ndarray, decisions: list[str], costs: dict[str, float]) -> float:
    total = 0.0
    for label, decision in zip(y_true, decisions):
        if decision == "approve" and label == 1:
            total += float(costs["fraud_miss_cost"])
        elif decision == "decline" and label == 0:
            total += float(costs["false_positive_cost"])
        elif decision == "review":
            total += float(costs["manual_review_cost"])
    return total / max(len(y_true), 1)


def summarize_risk_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    decisions: list[str],
    costs: dict[str, float],
) -> dict[str, Any]:
    y_true = frame["is_fraud"].astype(int).to_numpy()
    flagged = np.asarray([decision != "approve" for decision in decisions], dtype=int)
    review_mask = np.asarray([decision == "review" for decision in decisions], dtype=int)
    decline_mask = np.asarray([decision == "decline" for decision in decisions], dtype=int)

    auc = 0.5
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, probabilities))

    return {
        "auc": auc,
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "fraud_capture": float(((flagged == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
        "false_positive_rate": float(((flagged == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "review_rate": float(review_mask.mean()),
        "decline_rate": float(decline_mask.mean()),
        "average_expected_cost": expected_cost(y_true, decisions, costs),
    }


def slice_metrics(frame: pd.DataFrame, probabilities: np.ndarray, decisions: list[str]) -> dict[str, Any]:
    amount_bucket = pd.cut(
        frame["amount"],
        bins=[-np.inf, 200, 1000, np.inf],
        labels=["low", "medium", "high"],
    )
    decorated = frame.assign(score=probabilities, decision=decisions, amount_bucket=amount_bucket)
    output: dict[str, Any] = {}
    for slice_name in ["type", "amount_bucket"]:
        summary = {}
        for value, group in decorated.groupby(slice_name, dropna=False):
            label = "unknown" if pd.isna(value) else str(value)
            summary[label] = {
                "rows": int(len(group)),
                "fraud_rate": float(group["is_fraud"].mean()),
                "review_rate": float((group["decision"] == "review").mean()),
                "decline_rate": float((group["decision"] == "decline").mean()),
                "mean_score": float(group["score"].mean()),
            }
        output[slice_name] = summary
    return output
