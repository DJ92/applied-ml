from __future__ import annotations

import numpy as np

from src.runtime import sample_row_from_encoder, vectorize_row


def test_vectorize_row_uses_encoder_maps() -> None:
    encoder = {
        "categorical_features": ["user_id", "product_id"],
        "dense_features": ["price", "discount"],
        "category_maps": {
            "user_id": {"u1": 1},
            "product_id": {"p1": 1},
        },
    }
    categorical, dense = vectorize_row({"user_id": "u1", "product_id": "p1", "price": 10.0, "discount": 0.2}, encoder)
    assert categorical.shape == (1, 2)
    assert dense.shape == (1, 2)
    assert np.allclose(dense, [[10.0, 0.2]])


def test_sample_row_from_encoder_returns_defaults() -> None:
    encoder = {
        "categorical_features": ["user_id"],
        "dense_features": ["price"],
        "category_maps": {"user_id": {"u1": 1}},
    }
    sample = sample_row_from_encoder(encoder)
    assert sample["user_id"] == "u1"
    assert sample["price"] == 0.0
