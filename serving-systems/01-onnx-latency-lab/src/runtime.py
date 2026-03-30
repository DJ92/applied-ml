from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ranking_project_root() -> Path:
    return Path(__file__).resolve().parents[3] / "ranking-systems" / "01-dcnv2-commerce-ranking"


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ranking_bundle(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def load_ranking_model(bundle: dict[str, Any]) -> torch.nn.Module:
    models_module = load_module(ranking_project_root() / "src" / "models.py", "ranking_models_runtime")
    model = models_module.build_model(
        name=bundle["model_name"],
        cardinalities=bundle["cardinalities"],
        dense_dim=bundle["dense_dim"],
        config=bundle["model_config"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def vectorize_row(row: dict[str, Any], encoder_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    categorical = []
    for feature in encoder_payload["categorical_features"]:
        mapping = encoder_payload["category_maps"][feature]
        categorical.append(mapping.get(str(row.get(feature, "__missing__")), 0))
    dense = [float(row.get(feature, 0.0)) for feature in encoder_payload["dense_features"]]
    return (
        np.asarray([categorical], dtype=np.int64),
        np.asarray([dense], dtype=np.float32),
    )


def sample_row_from_encoder(encoder_payload: dict[str, Any]) -> dict[str, Any]:
    sample = {}
    for feature in encoder_payload["categorical_features"]:
        values = list(encoder_payload["category_maps"][feature].keys())
        sample[feature] = values[0] if values else "__missing__"
    for feature in encoder_payload["dense_features"]:
        sample[feature] = 0.0
    return sample


def sample_batch(bundle: dict[str, Any], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    encoder = bundle["encoder"]
    categorical = []
    for feature in encoder["categorical_features"]:
        max_id = max(encoder["category_maps"][feature].values(), default=0)
        categorical.append(np.random.randint(1, max_id + 1 if max_id > 0 else 1, size=batch_size))
    cat = np.stack(categorical, axis=1) if categorical else np.zeros((batch_size, 0), dtype=np.int64)
    dense = np.random.randn(batch_size, len(encoder["dense_features"])).astype(np.float32)
    return cat, dense


def metadata_path_for(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_suffix(model_path.suffix + ".metadata.json")


def write_metadata(model_path: str | Path, payload: dict[str, Any]) -> Path:
    path = metadata_path_for(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_metadata(model_path: str | Path) -> dict[str, Any]:
    return json.loads(metadata_path_for(model_path).read_text(encoding="utf-8"))
