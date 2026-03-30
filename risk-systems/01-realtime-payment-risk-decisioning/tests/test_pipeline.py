from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def write_raw_dataset(path: Path) -> None:
    rows = []
    event_id = 0
    for step in range(1, 61):
        for customer_idx in range(2):
            amount = float(80 + (step % 7) * 25 + customer_idx * 10)
            is_fraud = int(step % 11 == 0 and customer_idx == 1)
            flagged = int(is_fraud and step % 22 == 0)
            if is_fraud:
                amount *= 18
            old_origin = float(3000 + customer_idx * 100 - step * 3)
            new_origin = old_origin - amount + (5 if is_fraud else 0)
            old_dest = float(1000 + step * 2)
            new_dest = old_dest + amount - (25 if is_fraud else 0)
            rows.append(
                {
                    "step": step,
                    "type": "TRANSFER" if step % 3 == 0 else "PAYMENT",
                    "amount": amount,
                    "nameOrig": f"C{customer_idx}",
                    "oldbalanceOrg": old_origin,
                    "newbalanceOrig": new_origin,
                    "nameDest": f"M{event_id % 5}",
                    "oldbalanceDest": old_dest,
                    "newbalanceDest": new_dest,
                    "isFraud": is_fraud,
                    "isFlaggedFraud": flagged,
                }
            )
            event_id += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def write_config(tmp_path: Path, raw_path: Path) -> Path:
    base_config_path = Path(__file__).resolve().parents[1] / "configs" / "risk_model.yaml"
    config = yaml.safe_load(base_config_path.read_text())
    config["data"]["raw_path"] = str(raw_path)
    config["data"]["processed_path"] = str(tmp_path / "processed.csv")
    config["training"]["checkpoint_dir"] = str(tmp_path / "artifacts")
    config["review_queue"]["analyst_capacity"] = 5
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_prepare_data_and_train_pipeline(tmp_path: Path) -> None:
    raw_path = tmp_path / "paysim.csv"
    write_raw_dataset(raw_path)
    config_path = write_config(tmp_path, raw_path)
    project_root = Path(__file__).resolve().parents[1]

    prepared = subprocess.run(
        ["python", "src/prepare_data.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    prepared_payload = json.loads(prepared.stdout)
    assert Path(prepared_payload["output_path"]).exists()
    assert set(prepared_payload["split_counts"]) == {"train", "val", "test"}

    trained = subprocess.run(
        ["python", "src/train.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    trained_payload = json.loads(trained.stdout)
    assert Path(trained_payload["checkpoint"]).exists()
    assert "auc" in trained_payload["test"]


def test_evaluate_and_stream_scoring(tmp_path: Path) -> None:
    raw_path = tmp_path / "paysim.csv"
    write_raw_dataset(raw_path)
    config_path = write_config(tmp_path, raw_path)
    project_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        ["python", "src/prepare_data.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    train = subprocess.run(
        ["python", "src/train.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    checkpoint = json.loads(train.stdout)["checkpoint"]

    evaluate = subprocess.run(
        ["python", "src/evaluate.py", "--config", str(config_path), "--model", checkpoint],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    evaluate_payload = json.loads(evaluate.stdout)
    assert "slice_metrics" in evaluate_payload
    assert "amount_bucket" in evaluate_payload["slice_metrics"]

    score_stream = subprocess.run(
        [
            "python",
            "src/score_stream.py",
            "--config",
            str(config_path),
            "--model",
            checkpoint,
            "--events",
            str(tmp_path / "processed.csv"),
            "--max-events",
            "12",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    stream_payload = json.loads(score_stream.stdout)
    assert stream_payload["events"] == 12
    assert {"p50", "p95", "p99"} <= set(stream_payload["latency_ms"])


def test_review_queue_simulation(tmp_path: Path) -> None:
    raw_path = tmp_path / "paysim.csv"
    write_raw_dataset(raw_path)
    config_path = write_config(tmp_path, raw_path)
    project_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        ["python", "src/prepare_data.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    train = subprocess.run(
        ["python", "src/train.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    checkpoint = json.loads(train.stdout)["checkpoint"]

    review = subprocess.run(
        ["python", "src/simulate_review_queue.py", "--config", str(config_path), "--model", checkpoint],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    review_payload = json.loads(review.stdout)
    assert "overflow" in review_payload
    assert review_payload["within_capacity"] <= 5
