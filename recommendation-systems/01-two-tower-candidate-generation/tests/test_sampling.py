from __future__ import annotations

import pandas as pd

from src.data import RetrievalConfig, sample_negatives


def test_negative_sampling_avoids_seen_items() -> None:
    frame = pd.DataFrame(
        [
            {"user_id": "u1", "product_id": "p1", "device_type": "mobile", "category_id": "c1", "query_length": 2.0, "user_recency_days": 1.0, "price": 10.0, "discount": 0.0, "product_age_days": 5.0, "click": 1},
            {"user_id": "u1", "product_id": "p2", "device_type": "mobile", "category_id": "c2", "query_length": 2.0, "user_recency_days": 1.0, "price": 11.0, "discount": 0.1, "product_age_days": 4.0, "click": 0},
            {"user_id": "u2", "product_id": "p3", "device_type": "desktop", "category_id": "c1", "query_length": 3.0, "user_recency_days": 2.0, "price": 13.0, "discount": 0.2, "product_age_days": 3.0, "click": 1},
            {"user_id": "u2", "product_id": "p4", "device_type": "desktop", "category_id": "c3", "query_length": 3.0, "user_recency_days": 2.0, "price": 12.0, "discount": 0.0, "product_age_days": 6.0, "click": 0},
        ]
    )
    config = RetrievalConfig(
        data={
            "user_id_column": "user_id",
            "item_id_column": "product_id",
            "target": "click",
            "negative_samples": 1,
            "user_categorical_features": ["user_id", "device_type"],
            "item_categorical_features": ["product_id", "category_id"],
            "user_dense_features": ["query_length", "user_recency_days"],
            "item_dense_features": ["price", "discount", "product_age_days"],
        },
        training={"seed": 42},
        model={"name": "two_tower"},
    )
    sampled = sample_negatives(frame, config, seed=42)
    negatives = sampled[sampled["click"] == 0]
    seen_pairs = {("u1", "p1"), ("u1", "p2"), ("u2", "p3"), ("u2", "p4")}
    assert all((row.user_id, row.product_id) not in seen_pairs for row in negatives.itertuples())
