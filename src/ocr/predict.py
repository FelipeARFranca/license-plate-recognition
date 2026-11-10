from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "cct-s-v2-global-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize a license plate using a pretrained Fast-Plate-OCR model."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a cropped license plate image.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Fast-Plate-OCR model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="ONNX Runtime device (default: cpu).",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Skip per-character confidence output.",
    )
    return parser.parse_args()


def format_confidences(plate: str, confidences: Any) -> None:
    if confidences is None:
        print("Character confidence: unavailable")
        return

    try:
        values = [float(value) for value in confidences]
    except (TypeError, ValueError):
        print(f"Character confidence: {confidences}")
        return

    if len(values) != len(plate):
        print(f"Character confidence values: {values}")
        return

    print("Character confidence:")
    for character, confidence in zip(plate, values):
        print(f"  {character}: {confidence:.4f}")


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()

    if not image_path.is_file():
        print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
        return 1

    try:
        from fast_plate_ocr import LicensePlateRecognizer
    except ImportError as exc:
        print(
            "ERROR: Fast-Plate-OCR is not installed in this environment.\n"
            'Install it with: uv add "fast-plate-ocr[onnx]"',
            file=sys.stderr,
        )
        raise exc

    print("=" * 72)
    print("FAST-PLATE-OCR — SINGLE IMAGE INFERENCE")
    print("=" * 72)
    print(f"Image:  {image_path}")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    print()

    print("Loading recognizer (the first run may download model files)...")
    load_start = time.perf_counter()

    try:
        recognizer = LicensePlateRecognizer(args.model, device=args.device)
    except Exception as exc:
        print(f"ERROR while loading the OCR model: {exc}", file=sys.stderr)
        return 2

    load_seconds = time.perf_counter() - load_start

    providers = getattr(recognizer, "providers", None)
    if providers is not None:
        print(f"Execution providers: {providers}")

    print(f"Recognizer initialization: {load_seconds:.3f} s")
    print("\nRunning recognition...")

    inference_start = time.perf_counter()
    try:
        prediction = recognizer.run_one(
            str(image_path),
            return_confidence=not args.no_confidence,
        )
    except Exception as exc:
        print(f"ERROR during inference: {exc}", file=sys.stderr)
        return 3

    inference_seconds = time.perf_counter() - inference_start

    print("\n" + "=" * 72)
    print("PREDICTION")
    print("=" * 72)
    print(f"Recognized plate: {prediction.plate}")

    if not args.no_confidence:
        format_confidences(
            prediction.plate,
            getattr(prediction, "char_probs", None),
        )

    region = getattr(prediction, "region", None)
    if region is not None:
        print(f"Predicted region: {region}")
        region_prob = getattr(prediction, "region_prob", None)
        if region_prob is not None:
            print(f"Region confidence: {float(region_prob):.4f}")

    print(f"\nInference time: {inference_seconds * 1000:.2f} ms")
    print("=" * 72)
    print("Note: initialization time is separate and may include model download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
