from pathlib import Path
import argparse

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_CONFIG = PROJECT_ROOT / "configs" / "detection.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detection"

MODEL_NAME = "yolo26n.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# Training configurations
# uv run --extra cu130 python src/detection/train.py --smoke-test
# uv run --extra cu130 python src/detection/train.py --quick-experiment
# uv run --extra cu130 python src/detection/train.py
TRAINING_MODES = {
    "smoke_test": {
        "epochs": 2,
        "fraction": 0.05,
        "imgsz": 640,
        "patience": 2,
        "name": "yolo26n_smoke_test",
    },
    "quick_experiment": {
        "epochs": 20,
        "fraction": 0.20,
        "imgsz": 512,
        "patience": 6,
        "name": "yolo26n_quick_experiment",
    },
    "full": {
        "epochs": 100,
        "fraction": 1.0,
        "imgsz": 640,
        "patience": 20,
        "name": "yolo26n_baseline",
    },
}


def train(mode: str = "full"):
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {mode}")

    config = TRAINING_MODES[mode]

    print("=" * 50)
    print("LICENSE PLATE DETECTION TRAINING")
    print("=" * 50)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Mode: {mode}")
    print(f"Epochs: {config['epochs']}")
    print(f"Dataset fraction: {config['fraction']}")
    print(f"Image size: {config['imgsz']}")
    print(f"Patience: {config['patience']}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATA_CONFIG),

        # Training configuration
        epochs=config["epochs"],
        fraction=config["fraction"],
        imgsz=config["imgsz"],

        # Performance
        batch=-1,
        workers=4,
        device=DEVICE,
        amp=torch.cuda.is_available(),

        # Reproducibility and training control
        patience=config["patience"],
        seed=42,

        # Output
        project=str(OUTPUT_DIR),
        name=config["name"],
        plots=True,
        save=True,
    )

    print("\nTraining completed!")
    print(f"Results directory: {results.save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the license plate detection model."
    )

    modes = parser.add_mutually_exclusive_group()

    modes.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small test to verify the training pipeline.",
    )

    modes.add_argument(
        "--quick-experiment",
        action="store_true",
        help="Run a shorter training experiment.",
    )

    args = parser.parse_args()

    if args.smoke_test:
        mode = "smoke_test"
    elif args.quick_experiment:
        mode = "quick_experiment"
    else:
        mode = "full"

    train(mode)


if __name__ == "__main__":
    main()
