from __future__ import annotations

import numpy as np

from src.metrics import auc_score, expected_calibration_error, log_loss, ndcg_at_k


def test_auc_score_prefers_better_rankings() -> None:
    labels = np.array([1, 0, 1, 0], dtype=np.float32)
    good_scores = np.array([0.9, 0.2, 0.8, 0.1], dtype=np.float32)
    bad_scores = np.array([0.1, 0.8, 0.2, 0.9], dtype=np.float32)
    assert auc_score(labels, good_scores) > auc_score(labels, bad_scores)


def test_log_loss_is_finite() -> None:
    labels = np.array([1, 0, 1], dtype=np.float32)
    probs = np.array([0.9, 0.1, 0.8], dtype=np.float32)
    assert log_loss(labels, probs) >= 0.0


def test_ndcg_uses_groups() -> None:
    labels = np.array([1, 0, 0, 1], dtype=np.float32)
    scores = np.array([0.9, 0.2, 0.1, 0.8], dtype=np.float32)
    groups = np.array(["a", "a", "b", "b"])
    assert ndcg_at_k(labels, scores, groups, k=2) > 0.9


def test_expected_calibration_error_bounds() -> None:
    labels = np.array([1, 0, 1, 0], dtype=np.float32)
    probs = np.array([0.8, 0.2, 0.7, 0.3], dtype=np.float32)
    ece = expected_calibration_error(labels, probs)
    assert 0.0 <= ece <= 1.0
