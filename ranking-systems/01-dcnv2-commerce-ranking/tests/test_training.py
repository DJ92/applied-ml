from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def write_dataset(path: Path) -> None:
    rows = []
    for session_idx in range(12):
        for product_idx in range(4):
            rows.append(
                {
                    "user_id": f"user-{session_idx % 4}",
                    "product_id": f"product-{product_idx}",
                    "category_id": f"category-{product_idx % 2}",
                    "device_type": "mobile" if session_idx % 2 == 0 else "desktop",
                    "price": float(product_idx + 1),
                    "discount": float(product_idx % 2) * 0.1,
                    "query_length": float(2 + product_idx),
                    "product_age_days": float(product_idx + 3),
                    "user_recency_days": float(session_idx % 5),
                    "click": 1 if product_idx == session_idx % 4 else 0,
                    "conversion": 1 if product_idx == 0 and session_idx % 3 == 0 else 0,
                    "session_id": f"session-{session_idx}",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_train_and_evaluate_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "commerce.csv"
    write_dataset(dataset_path)

    base_config_path = Path(__file__).resolve().parents[1] / "configs" / "dcnv2.yaml"
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
    payload = json.loads(train.stdout)
    checkpoint_path = payload["checkpoint"]
    assert Path(checkpoint_path).exists()

    evaluate = subprocess.run(
        ["python", "src/evaluate.py", "--config", str(config_path), "--model", checkpoint_path],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    eval_payload = json.loads(evaluate.stdout)
    assert "auc" in eval_payload
    assert "ndcg@10" in eval_payload
