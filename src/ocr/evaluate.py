from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_MODEL = "cct-s-v2-global-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained Fast-Plate-OCR model against a labeled CSV."
    )
    parser.add_argument("--root", type=Path, default=Path("datasets/ocr/plates"),
                        help="Dataset directory containing plates.csv and images.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV path; default: <root>/plates.csv.")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Image directory; default: <root>/plate_imgs.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name; default: {DEFAULT_MODEL}.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="ONNX Runtime device; default: cpu.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ocr/evaluation"),
                        help="Directory for prediction CSV and summary JSON.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional row limit for a quick smoke test.")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warm-up inference.")
    return parser.parse_args()


def resolve_project_path(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def normalize_text(text: str) -> str:
    """Case- and whitespace-normalization only; punctuation is preserved."""
    return "".join(text.upper().split())


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
        previous = current
    return previous[-1]


def build_image_index(root: Path):
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            resolved = path.resolve()
            by_name[path.name.casefold()].append(resolved)
            by_stem[path.stem.casefold()].append(resolved)
    return by_name, by_stem


def resolve_image(filename: str, image_dir: Path, by_name, by_stem):
    supplied = Path(filename.replace("\\", "/"))
    basename = supplied.name
    expected = (image_dir / basename).resolve()

    if expected.is_file():
        return expected, "expected_path"

    exact = sorted(set(by_name.get(basename.casefold(), [])))
    if len(exact) == 1:
        return exact[0], "basename"
    if len(exact) > 1:
        return None, "ambiguous_basename"

    same_stem = sorted(set(by_stem.get(supplied.stem.casefold(), [])))
    if len(same_stem) == 1:
        return same_stem[0], "stem_extension_mismatch"
    if len(same_stem) > 1:
        return None, "ambiguous_stem"
    return None, "missing"


def load_annotations(csv_path: Path, dataset_root: Path,
                     image_dir: Path, limit: int | None):
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    by_name, by_stem = build_image_index(dataset_root)
    rows = []
    resolution_counts: dict[str, int] = defaultdict(int)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"filename", "plate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; found {reader.fieldnames}."
            )

        for row_number, row in enumerate(reader, start=2):
            if limit is not None and len(rows) >= limit:
                break
            filename = (row.get("filename") or "").strip()
            label = (row.get("plate") or "").strip()

            if not filename:
                status, image_path = "empty_filename", None
            elif not label:
                status, image_path = "empty_label", None
            else:
                image_path, status = resolve_image(
                    filename, image_dir, by_name, by_stem
                )

            resolution_counts[status] += 1
            rows.append({
                "csv_row": row_number,
                "filename": filename,
                "ground_truth": label,
                "image_path": image_path,
                "resolution": status,
            })

    return rows, dict(resolution_counts)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def prediction_text(prediction: Any) -> str:
    value = getattr(prediction, "plate", None)
    if value is None:
        raise ValueError("Prediction has no 'plate' field.")
    return str(value)


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    # This file lives at <project>/src/ocr/evaluate.py.
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = resolve_project_path(args.root, project_root)
    csv_path = (resolve_project_path(args.csv, project_root)
                if args.csv else dataset_root / "plates.csv")
    image_dir = (resolve_project_path(args.image_dir, project_root)
                 if args.image_dir else dataset_root / "plate_imgs")
    output_dir = resolve_project_path(args.output_dir, project_root)

    print("=" * 76)
    print("BRAZILIAN LICENSE PLATE OCR — COMPLETE EVALUATION")
    print("=" * 76)
    print(f"Dataset root: {dataset_root}")
    print(f"CSV:          {csv_path}")
    print(f"Images:       {image_dir}")
    print(f"Model:        {args.model}")
    print(f"Device:       {args.device}")

    try:
        rows, resolution_counts = load_annotations(
            csv_path, dataset_root, image_dir, args.limit
        )
    except Exception as exc:
        print(f"ERROR loading dataset: {exc}", file=sys.stderr)
        return 1

    resolved_rows = [r for r in rows if r["image_path"] is not None]
    unresolved_rows = [r for r in rows if r["image_path"] is None]

    print("\\nDATASET PREFLIGHT")
    print("-" * 76)
    print(f"CSV rows:          {len(rows)}")
    print(f"Resolved images:   {len(resolved_rows)}")
    print(f"Unresolved rows:   {len(unresolved_rows)}")
    for status, count in sorted(resolution_counts.items()):
        print(f"  {status}: {count}")

    if not resolved_rows:
        print("No image references resolved. Check --root/--csv/--image-dir.",
              file=sys.stderr)
        return 2

    if unresolved_rows:
        print("\\nUnresolved examples:")
        for row in unresolved_rows[:10]:
            print(f"  row {row['csv_row']}: {row['filename']!r} ({row['resolution']})")

    try:
        from fast_plate_ocr import LicensePlateRecognizer
    except ImportError as exc:
        print('Install Fast-Plate-OCR with: uv add "fast-plate-ocr[onnx]"',
              file=sys.stderr)
        raise exc

    print("\\nLoading recognizer...")
    load_start = time.perf_counter()
    try:
        recognizer = LicensePlateRecognizer(args.model, device=args.device)
    except Exception as exc:
        print(f"ERROR loading model: {exc}", file=sys.stderr)
        return 3
    load_seconds = time.perf_counter() - load_start
    print(f"Initialization: {load_seconds:.3f} s")

    warmup_ms = None
    if not args.no_warmup:
        try:
            start = time.perf_counter()
            recognizer.run_one(str(resolved_rows[0]["image_path"]))
            warmup_ms = (time.perf_counter() - start) * 1000
            print(f"Warm-up: {warmup_ms:.2f} ms")
        except Exception as exc:
            print(f"ERROR during warm-up: {exc}", file=sys.stderr)
            return 4

    print("\\nRunning inference...")
    results = []
    latencies_ms = []
    edit_total = 0
    gt_char_total = 0
    exact_matches = 0
    inference_errors = 0

    for index, row in enumerate(resolved_rows, start=1):
        start = time.perf_counter()
        try:
            pred_obj = recognizer.run_one(str(row["image_path"]))
            prediction = prediction_text(pred_obj)
            latency = (time.perf_counter() - start) * 1000
            gt_norm = normalize_text(row["ground_truth"])
            pred_norm = normalize_text(prediction)
            distance = edit_distance(gt_norm, pred_norm)
            is_exact = gt_norm == pred_norm

            latencies_ms.append(latency)
            edit_total += distance
            gt_char_total += len(gt_norm)
            exact_matches += int(is_exact)

            results.append({
                "csv_row": row["csv_row"],
                "filename": row["filename"],
                "resolved_image_path": str(row["image_path"]),
                "ground_truth": row["ground_truth"],
                "prediction": prediction,
                "exact_match": is_exact,
                "edit_distance": distance,
                "cer": distance / len(gt_norm) if gt_norm else "",
                "inference_ms": round(latency, 3),
                "error": "",
            })
        except Exception as exc:
            inference_errors += 1
            results.append({
                "csv_row": row["csv_row"],
                "filename": row["filename"],
                "resolved_image_path": str(row["image_path"]),
                "ground_truth": row["ground_truth"],
                "prediction": "",
                "exact_match": False,
                "edit_distance": "",
                "cer": "",
                "inference_ms": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

        if index % 10 == 0 or index == len(resolved_rows):
            print(f"Processed {index}/{len(resolved_rows)} images...")

    successful = len(latencies_ms)
    accuracy = exact_matches / successful if successful else 0.0
    cer = edit_total / gt_char_total if gt_char_total else None
    total_inference_seconds = sum(latencies_ms) / 1000

    summary = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "csv": str(csv_path),
        "image_directory": str(image_dir),
        "model": args.model,
        "device": args.device,
        "text_normalization": "uppercase and remove whitespace; preserve punctuation",
        "counts": {
            "csv_rows": len(rows),
            "resolved_images": len(resolved_rows),
            "unresolved_rows": len(unresolved_rows),
            "successful_inferences": successful,
            "inference_errors": inference_errors,
            "exact_matches": exact_matches,
        },
        "metrics": {
            "exact_plate_accuracy": accuracy,
            "exact_plate_accuracy_percent": accuracy * 100,
            "character_error_rate": cer,
            "character_error_rate_percent": cer * 100 if cer is not None else None,
            "total_edit_distance": edit_total,
            "total_ground_truth_characters": gt_char_total,
        },
        "latency_ms": {
            "mean": statistics.mean(latencies_ms) if successful else None,
            "median": statistics.median(latencies_ms) if successful else None,
            "p95": percentile(latencies_ms, 0.95),
            "min": min(latencies_ms) if successful else None,
            "max": max(latencies_ms) if successful else None,
            "total_inference_seconds": total_inference_seconds,
            "model_initialization_seconds": load_seconds,
            "warmup_ms": warmup_ms,
        },
        "image_resolution_methods": resolution_counts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "brazilian_plates_predictions.csv"
    summary_json = output_dir / "brazilian_plates_summary.json"

    fields = [
        "csv_row", "filename", "resolved_image_path", "ground_truth",
        "prediction", "exact_match", "edit_distance", "cer",
        "inference_ms", "error",
    ]
    with results_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\\n" + "=" * 76)
    print("EVALUATION SUMMARY")
    print("=" * 76)
    print(f"CSV rows:                 {len(rows)}")
    print(f"Resolved images:          {len(resolved_rows)}")
    print(f"Successful inferences:    {successful}")
    print(f"Unresolved rows:          {len(unresolved_rows)}")
    print(f"Inference errors:         {inference_errors}")
    print(f"Exact matches:            {exact_matches}/{successful}")
    print(f"Exact plate accuracy:     {accuracy * 100:.2f}%")
    print(f"Character error rate:     {cer * 100:.2f}%" if cer is not None
          else "Character error rate:     unavailable")
    if successful:
        print(f"Mean inference time:      {statistics.mean(latencies_ms):.2f} ms")
        print(f"Median inference time:    {statistics.median(latencies_ms):.2f} ms")
        print(f"P95 inference time:       {percentile(latencies_ms, 0.95):.2f} ms")
        print(f"Total inference time:     {total_inference_seconds:.2f} s")
    print(f"\\nPredictions: {results_csv}")
    print(f"Summary:     {summary_json}")

    if unresolved_rows or inference_errors:
        print("\\nWARNING: Some samples were not evaluated. Review the CSV before "
              "interpreting accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
