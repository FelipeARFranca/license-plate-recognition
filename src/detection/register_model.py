
from pathlib import Path
import shutil

import bentoml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "detection"
    / "yolo26n_baseline"
)

MODEL_PATH = RUN_DIR / "weights" / "best.pt"

MODEL_NAME = "license_plate_detector"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    metadata = {
        "architecture": "YOLO26n",
        "framework": "Ultralytics",
        "task": "object_detection",
        "dataset": "Large License Plate Detection Dataset",
        "image_size": 640,
        "classes": {
            "0": "license_plate"
        },
        "test_metrics": {
            "precision": 0.9261,
            "recall": 0.8814,
            "mAP50": 0.9142,
            "mAP75": 0.7982,
            "mAP50_95": 0.6795,
        },
    }

    labels = {
        "task": "object-detection",
        "framework": "ultralytics",
        "stage": "baseline",
    }

    print(f"Registering model: {MODEL_NAME}")
    print(f"Source checkpoint: {MODEL_PATH}")

    with bentoml.models.create(
        name=MODEL_NAME,
        labels=labels,
        metadata=metadata,
    ) as model_ref:

        model_store_path = Path(model_ref.path)

        shutil.copy2(
            MODEL_PATH,
            model_store_path / "best.pt",
        )

        # Preserve additional training artifacts.
        for filename in ("results.csv", "args.yaml"):
            artifact = RUN_DIR / filename

            if artifact.is_file():
                shutil.copy2(
                    artifact,
                    model_store_path / filename,
                )

    print("\nModel successfully registered!")
    print(f"Model tag: {model_ref.tag}")
    print(f"Model location: {model_ref.path}")


if __name__ == "__main__":
    main()
