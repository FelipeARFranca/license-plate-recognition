from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODEL_NAME = "cct-s-v2-global-model"
BENTOML_MODEL_NAME = "license_plate_ocr"


def main() -> int:
    try:
        import bentoml
        from fast_plate_ocr.inference.hub import download_model
    except ImportError as exc:
        print(
            "Missing dependency. Ensure BentoML and Fast-Plate-OCR are installed "
            'in this uv environment (e.g. uv add bentoml "fast-plate-ocr[onnx]").',
            file=sys.stderr,
        )
        raise exc

    print("=" * 72)
    print("FAST-PLATE-OCR — BENTOML MODEL REGISTRATION")
    print("=" * 72)
    print(f"Model Hub identifier: {MODEL_NAME}")
    print(f"BentoML model name:   {BENTOML_MODEL_NAME}")
    print("\nResolving pretrained ONNX model and matching plate configuration...")

    try:
        onnx_path, config_path = download_model(MODEL_NAME)
    except Exception as exc:
        print(f"ERROR downloading/resolving model files: {exc}", file=sys.stderr)
        return 1

    onnx_path = Path(onnx_path)
    config_path = Path(config_path)

    if not onnx_path.is_file() or not config_path.is_file():
        print(
            f"Model files were not found:\nONNX: {onnx_path}\nConfig: {config_path}",
            file=sys.stderr,
        )
        return 2

    print(f"ONNX model: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"Plate config: {config_path}")

    metadata = {
        "task": "license_plate_recognition",
        "source": "Fast-Plate-OCR Model Hub",
        "source_model": MODEL_NAME,
        "architecture": "Compact Convolutional Transformer (CCT)",
        "inference_format": "ONNX",
        "inference_runtime": "ONNX Runtime",
        "plate_config": "plate_config.yaml",
        "benchmark": {
            "dataset": "Brazilian License Plate OCR (Kaggle: luisasr2)",
            "sample_count": 206,
            "exact_plate_accuracy": 0.9417,
            "character_error_rate": 0.0146,
            "mean_inference_ms": 25.46,
            "median_inference_ms": 22.91,
            "p95_inference_ms": 38.95,
        },
    }

    try:
        with bentoml.models.create(
            name=BENTOML_MODEL_NAME,
            labels={
                "task": "license_plate_recognition",
                "architecture": "cct",
                "source": "fast_plate_ocr",
            },
            metadata=metadata,
        ) as model_ref:
            destination_model = Path(model_ref.path_of("model.onnx"))
            destination_config = Path(model_ref.path_of("plate_config.yaml"))

            shutil.copy2(onnx_path, destination_model)
            shutil.copy2(config_path, destination_config)

        print("\n" + "=" * 72)
        print("REGISTRATION COMPLETED")
        print("=" * 72)
        print(f"Model tag: {model_ref.tag}")
        print(f"Stored ONNX: {destination_model}")
        print(f"Stored config: {destination_config}")
        print("\nBoth files are now stored together in BentoML Model Store.")
        return 0
    except Exception as exc:
        print(f"ERROR registering model in BentoML: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
