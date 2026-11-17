"""Command-line client for the license plate recognition pipeline service.

Sends a vehicle image to a running ``LicensePlatePipeline`` BentoML server
(source service, built Bento, or Docker container) and saves the returned
plate crops, annotated image, and JSON metadata.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://localhost:3000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="license-plate-recognition",
        description=(
            "Recognize license plates in an image using a running "
            "license plate pipeline service."
        ),
    )
    parser.add_argument("image", type=Path, help="Path to a vehicle image.")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Pipeline service URL (default: {DEFAULT_URL}).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Detector confidence threshold, 0-1 (default: 0.25).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Detector NMS IoU threshold, 0-1 (default: 0.7).",
    )
    parser.add_argument(
        "--no-ocr-confidence",
        action="store_true",
        help="Do not request per-character OCR confidences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the annotated image, plate crops, and result JSON "
            "(default: runs/pipeline/<image name>)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60).",
    )
    return parser.parse_args(argv)


def save_png_payload(payload: dict[str, Any], destination: Path) -> None:
    destination.write_bytes(base64.b64decode(payload["data"]))


def save_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write images to disk and return the result with Base64 data replaced by paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(result)

    annotated = result.get("annotated_image")
    if annotated:
        annotated_path = output_dir / "annotated.png"
        save_png_payload(annotated, annotated_path)
        summary["annotated_image"] = str(annotated_path)

    detections = []
    for detection in result.get("detections", []):
        detection = dict(detection)
        crop = detection.get("cropped_image")
        if crop:
            crop_path = output_dir / f"plate_{detection.get('id')}.png"
            save_png_payload(crop, crop_path)
            detection["cropped_image"] = str(crop_path)
        detections.append(detection)
    summary["detections"] = detections

    with (output_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return summary


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print(f"Plates detected:        {summary.get('detection_count', 0)}")
    print(f"Successful recognition: {summary.get('successful_recognitions', 0)}")

    for detection in summary.get("detections", []):
        text = detection.get("plate_text") or "<unrecognized>"
        confidence = detection.get("detection_confidence", 0.0)
        bbox = detection.get("bbox_xyxy")
        print(f"  #{detection.get('id')}: {text}  "
              f"(detection {confidence:.2f}, bbox {bbox})")
        for key in ("crop_error", "ocr_error"):
            if detection.get(key):
                print(f"      {key}: {detection[key]}")

    total_ms = summary.get("timings_ms", {}).get("total")
    if total_ms is not None:
        print(f"Server processing time: {total_ms:.1f} ms")
    print(f"Outputs saved to:       {output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
        return 1

    for name in ("confidence_threshold", "iou_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            print(f"ERROR: --{name.replace('_', '-')} must be between 0 and 1.",
                  file=sys.stderr)
            return 1

    output_dir = (args.output_dir or Path("runs") / "pipeline" / image_path.stem)
    output_dir = output_dir.expanduser().resolve()

    # Imported lazily so `--help` and argument errors stay fast.
    import bentoml

    try:
        with bentoml.SyncHTTPClient(args.url, timeout=args.timeout) as client:
            result = client.recognize(
                image=image_path,
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                return_ocr_confidence=not args.no_ocr_confidence,
            )
    except Exception as exc:
        print(
            f"ERROR: Request to the pipeline service at {args.url} failed: "
            f"{type(exc).__name__}: {exc}\n"
            "Make sure the service is running, e.g.:\n"
            "  uv run --extra cpu bentoml serve "
            "services.pipeline_service:LicensePlatePipeline",
            file=sys.stderr,
        )
        return 2

    summary = save_outputs(result, output_dir)
    print_summary(summary, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
