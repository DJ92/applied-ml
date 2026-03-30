from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    pos_count = positives.sum()
    neg_count = negatives.sum()
    if pos_count == 0 or neg_count == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[positives].sum()
    return float((pos_ranks - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count))


def log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
    clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= start) & (probs < end if end < 1 else probs <= end)
        if not np.any(mask):
            continue
        avg_conf = probs[mask].mean()
        avg_true = y_true[mask].mean()
        ece += np.abs(avg_conf - avg_true) * mask.mean()
    return float(ece)


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    unique_groups = np.unique(groups)
    ndcgs = []
    for group in unique_groups:
        mask = groups == group
        if mask.sum() <= 1:
            continue
        labels = y_true[mask]
        scores = y_score[mask]
        order = np.argsort(scores)[::-1][:k]
        ranked = labels[order]
        gains = (2 ** ranked - 1) / np.log2(np.arange(2, len(ranked) + 2))
        dcg = gains.sum()
        ideal = np.sort(labels)[::-1][:k]
        ideal_gains = (2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))
        idcg = ideal_gains.sum()
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def inference_latency_ms(model: torch.nn.Module, categorical: np.ndarray, dense: np.ndarray, runs: int = 25) -> float:
    if len(categorical) == 0:
        return 0.0
    sample_cat = torch.as_tensor(categorical[: min(len(categorical), 32)], dtype=torch.long)
    sample_dense = torch.as_tensor(dense[: min(len(dense), 32)], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_cat, sample_dense)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(sample_cat, sample_dense)
        elapsed = (time.perf_counter() - start) / runs
    return float(elapsed * 1000.0)


def summarize_ranking_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    groups: np.ndarray,
    model: torch.nn.Module | None = None,
    categorical: np.ndarray | None = None,
    dense: np.ndarray | None = None,
) -> dict[str, Any]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    summary: dict[str, Any] = {
        "auc": auc_score(y_true, probs),
        "log_loss": log_loss(y_true, probs),
        "ndcg@10": ndcg_at_k(y_true, probs, groups, k=10),
        "ece": expected_calibration_error(y_true, probs),
    }
    if model is not None and categorical is not None and dense is not None:
        summary["cpu_latency_ms"] = inference_latency_ms(model, categorical, dense)
        summary["parameter_count"] = int(sum(param.numel() for param in model.parameters()))
    return summary
