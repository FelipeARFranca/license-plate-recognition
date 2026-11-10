from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detection"
    / "yolo26n_baseline"
    / "weights"
    / "best.pt"
)

DATA_YAML = PROJECT_ROOT / "configs" / "detection.yaml"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    print(f"Loading model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Starting evaluation on the test split...")

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "detection"),
        name="yolo26n_test",
    )

    print("\n===== DETECTION TEST RESULTS =====")
    print(f"Precision:   {results.box.mp:.4f}")
    print(f"Recall:      {results.box.mr:.4f}")
    print(f"mAP50:       {results.box.map50:.4f}")
    print(f"mAP75:       {results.box.map75:.4f}")
    print(f"mAP50-95:    {results.box.map:.4f}")

    print("\nInference speed:")
    print(f"{results.speed}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
