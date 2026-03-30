from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from src.index import NumpyANNIndex


def write_dataset(path: Path) -> None:
    rows = []
    for user_idx in range(6):
        for item_idx in range(8):
            rows.append(
                {
                    "user_id": f"user-{user_idx}",
                    "product_id": f"item-{item_idx}",
                    "category_id": f"category-{item_idx % 3}",
                    "device_type": "mobile" if user_idx % 2 == 0 else "desktop",
                    "query_length": float((item_idx % 4) + 1),
                    "user_recency_days": float(user_idx % 3),
                    "price": float(item_idx + 1),
                    "discount": float(item_idx % 2) * 0.1,
                    "product_age_days": float(item_idx + 2),
                    "click": 1 if item_idx == (user_idx + 1) % 8 or item_idx == user_idx % 8 else 0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_numpy_index_returns_topk_shapes() -> None:
    index = NumpyANNIndex(["a", "b", "c"], item_vectors=[[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    items, scores = index.search(query_vectors=[[1.0, 0.0]], topk=2)
    assert items.shape == (1, 2)
    assert scores.shape == (1, 2)


def test_retrieval_pipeline_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "commerce.csv"
    write_dataset(dataset_path)

    base_config_path = Path(__file__).resolve().parents[1] / "configs" / "two_tower.yaml"
    config = yaml.safe_load(base_config_path.read_text())
    config["data"]["path"] = str(dataset_path)
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 16
    config["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    project_root = Path(__file__).resolve().parents[1]
    train = subprocess.run(
        ["python", "src/train.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    checkpoint_path = json.loads(train.stdout)["checkpoint"]

    build_index = subprocess.run(
        ["python", "src/build_index.py", "--checkpoint", checkpoint_path, "--output", str(tmp_path / "index.pkl")],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    index_path = json.loads(build_index.stdout)["index_path"]
    assert Path(index_path).exists()

    evaluate = subprocess.run(
        ["python", "src/evaluate.py", "--checkpoint", checkpoint_path, "--index", index_path, "--topk", "5,10"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    metrics = json.loads(evaluate.stdout)
    assert "recall@5" in metrics
