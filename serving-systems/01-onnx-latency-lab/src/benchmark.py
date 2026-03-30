from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import load_metadata, load_ranking_bundle, load_ranking_model, metadata_path_for, sample_batch


def benchmark_callable(fn, runs: int = 50) -> dict[str, float]:
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "throughput_qps": float(1000.0 / max(arr.mean(), 1e-6)),
    }


def ensure_onnx_export(checkpoint: str, output_path: Path) -> Path:
    metadata_path = metadata_path_for(output_path)
    if output_path.exists() and metadata_path.exists():
        return output_path
    from src.export import main as export_main  # local import to avoid hard dependency during pure PyTorch runs

    raise RuntimeError(
        "ONNX artifact missing. Run `python src/export.py --checkpoint ... --output ...` before ONNX benchmarking."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNX inference paths.")
    parser.add_argument(
        "--checkpoint",
        default="../../ranking-systems/01-dcnv2-commerce-ranking/checkpoints/dcnv2.pt",
    )
    parser.add_argument("--onnx-model", default="artifacts/dcnv2.onnx")
    parser.add_argument("--engine", default="pytorch,onnx")
    parser.add_argument("--batch-sizes", default="1,8,32")
    args = parser.parse_args()

    bundle = load_ranking_bundle(args.checkpoint)
    results: dict[str, dict[str, dict[str, float] | str]] = {}

    if "pytorch" in args.engine.split(","):
        model = load_ranking_model(bundle)
        results["pytorch"] = {}
        for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
            categorical, dense = sample_batch(bundle, batch_size)
            cat_tensor = torch.as_tensor(categorical, dtype=torch.long)
            dense_tensor = torch.as_tensor(dense, dtype=torch.float32)
            with torch.no_grad():
                results["pytorch"][str(batch_size)] = benchmark_callable(lambda: model(cat_tensor, dense_tensor))

    if "onnx" in args.engine.split(","):
        try:
            import onnxruntime as ort

            onnx_path = Path(args.onnx_model)
            ensure_onnx_export(args.checkpoint, onnx_path)
            _ = load_metadata(onnx_path)
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            results["onnx"] = {}
            for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
                categorical, dense = sample_batch(bundle, batch_size)
                payload = {
                    "categorical_features": categorical.astype("int64"),
                    "dense_features": dense.astype("float32"),
                }
                results["onnx"][str(batch_size)] = benchmark_callable(lambda: session.run(None, payload))
        except Exception as exc:
            results["onnx"] = {"status": f"skipped: {exc}"}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
