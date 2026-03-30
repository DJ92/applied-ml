from __future__ import annotations

import pytest

from src.feature_store import MockFeatureStore


def test_feature_store_uses_cache() -> None:
    store = MockFeatureStore(records={"user:u1": {"value": 1}}, lookup_latency_ms=0.0)
    assert store.get("user:u1")["value"] == 1
    assert store.get("user:u1")["value"] == 1
    assert store.stats()["cache_hits"] == 1


def test_feature_store_times_out() -> None:
    store = MockFeatureStore(records={"user:u1": {"value": 1}}, lookup_latency_ms=50.0, timeout_ms=10.0)
    with pytest.raises(TimeoutError):
        store.get("user:u1")
