from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import load_ranking_bundle, load_ranking_model, sample_batch, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a ranking checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="../../ranking-systems/01-dcnv2-commerce-ranking/checkpoints/dcnv2.pt",
    )
    parser.add_argument("--output", default="artifacts/dcnv2.onnx")
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("ONNX export requires the `onnx` package. Install requirements.txt first.") from exc

    bundle = load_ranking_bundle(args.checkpoint)
    model = load_ranking_model(bundle)
    categorical, dense = sample_batch(bundle, batch_size=1)
    torch.onnx.export(
        model,
        (
            torch.as_tensor(categorical, dtype=torch.long),
            torch.as_tensor(dense, dtype=torch.float32),
        ),
        args.output,
        input_names=["categorical_features", "dense_features"],
        output_names=["logits"],
        dynamic_axes={
            "categorical_features": {0: "batch"},
            "dense_features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    write_metadata(
        args.output,
        {
            "checkpoint": args.checkpoint,
            "encoder": bundle["encoder"],
            "model_name": bundle["model_name"],
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
