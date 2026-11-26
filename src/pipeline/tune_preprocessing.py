"""Evaluation and hyperparameter search for the low-light preprocessing filter.

Runs the detection + OCR pipeline in-process (models loaded once from the
BentoML Model Store) and uses Optuna to search gamma correction and CLAHE
parameters together with the detector's confidence threshold and resolution.

Ground truth (datasets/absolute_test)
  * Boxes: YOLO labels in ``anotated_yolo/labels``, one line per plate.
  * Text:  the ``"text"`` field of the JSON next to each image in
    ``cropped_images``, paired with the YOLO boxes by IoU.
  Every box counts for detection. Plates whose text is empty or is not a
  complete Brazilian plate (e.g. cut off at the image edge) are left out of
  the OCR and end-to-end metrics.

Metrics, reported separately so each model can be judged on its own
  * Detection: precision, recall and F1 at the confidence threshold, and AP50
    (box matching at IoU 0.5).
  * OCR: the recognizer run on the ground-truth crops, so detection errors
    do not count against it.
  * End to end: a plate is correct when a detection matches its box and the
    OCR text of that detection equals the annotation.
  The search maximizes end-to-end plate accuracy, with detection F1 and
  character error rate as small tie-breakers.

Evaluation design (small dataset)
  * Nested cross-validation on the night images, split by image: each outer
    fold is scored by parameters tuned without it (unbiased estimate).
  * A final search on all night images gives the parameters to deploy.
  * Daylight checks: the daylight test images and the detection test split,
    with the filter gated by a darkness threshold.

Visual outputs (--visuals): for every trial, the filtered frames, annotated
frames (ground truth vs predictions) and a sheet of plate crops before/after
the filter with ground-truth and recognized text.

Use --evaluate baseline|<best_params.json> to evaluate one configuration on
all images without searching.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr.evaluate import edit_distance  # noqa: E402
from services.pipeline_service import clamp_bbox  # noqa: E402
from services.preprocessing import FilterParams, apply_filter, mean_lightness  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_EXCLUDE = (
    "PCY9H63-TrafficRunRedLight-20260926052054.jpeg",
    "RZI2A08-TrafficRunRedLight-20260928133302.jpeg",
)
# Old (ABC1234) and Mercosul (ABC1D23) Brazilian plates.
PLATE_PATTERN = re.compile(r"[A-Z]{3}[0-9][A-Z0-9][0-9]{2}")

IOU_THRESHOLD = 0.7      # detector NMS, same as the services
MATCH_IOU = 0.5          # prediction <-> ground-truth box matching
TEXT_PAIR_IOU = 0.9      # YOLO box <-> JSON box pairing (same annotation)
DETECTION_FLOOR = 0.01   # detect once at this confidence; thresholds applied after
BATCH_SIZE = 4           # images decoded and detected at once
BASELINE_CONF = 0.25
BASELINE_IMGSZ = 640

CONF_RANGE = (0.05, 0.50)
GAMMA_RANGE = (0.30, 1.00)
TARGET_LIGHTNESS_RANGE = (0.25, 0.60)
CLAHE_CLIP_RANGE = (0.5, 10.0)   # log scale
CLAHE_TILE_RANGE = (2, 32)       # tile grid is N x N


def normalize_plate(text: str) -> str:
    """Uppercase and keep only letters and digits, so 'abc-1d23' == 'ABC1D23'."""
    return "".join(char for char in str(text).upper() if char.isalnum())


# --------------------------------------------------------------------------
# Configuration and search space
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    filter: FilterParams
    conf: float = BASELINE_CONF
    imgsz: int = BASELINE_IMGSZ
    enhance_crops: bool = False   # OCR reads crops from the filtered image

    def without_filter(self) -> "PipelineConfig":
        return replace(self, filter=FilterParams(), enhance_crops=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["filter"] = self.filter.to_dict()
        return data

    def describe(self) -> str:
        f = self.filter
        parts = []
        if f.gamma_mode == "fixed":
            parts.append(f"gamma {f.gamma:.2f}")
        elif f.gamma_mode == "adaptive":
            parts.append(f"adaptive gamma ->{f.target_lightness:.2f}")
        if f.clahe:
            parts.append(f"CLAHE clip {f.clahe_clip:.2f} tile {f.clahe_tile}")
        if f.gamma_mode != "off" and f.clahe:
            parts.append(f.order)
        if not f.is_identity:
            parts.append(f.channel.upper())
            parts.append("OCR on filtered crops" if self.enhance_crops
                         else "OCR on original crops")
        filt = ", ".join(parts) if parts else "no filter"
        return f"{filt} | conf {self.conf:.2f} | imgsz {self.imgsz}"


BASELINE = PipelineConfig(filter=FilterParams())

BASELINE_TRIAL = {
    "gamma_mode": "off", "clahe": False,
    "conf": BASELINE_CONF, "imgsz": BASELINE_IMGSZ,
}
# The filter tested before the search, kept as a reference point.
PREVIOUS_FILTER_TRIAL = {
    "gamma_mode": "adaptive", "target_lightness": 0.45,
    "clahe": True, "clahe_clip": 2.0, "clahe_tile": 8,
    "order": "gamma_first", "channel": "lab", "enhance_crops": False,
    "conf": BASELINE_CONF, "imgsz": BASELINE_IMGSZ,
}


def config_from_params(params: dict[str, Any]) -> PipelineConfig:
    """Rebuild a PipelineConfig from Optuna's flat parameter dictionary."""
    filter_params = FilterParams(
        gamma_mode=params.get("gamma_mode", "off"),
        gamma=float(params.get("gamma", 1.0)),
        target_lightness=float(params.get("target_lightness", 0.45)),
        clahe=bool(params.get("clahe", False)),
        clahe_clip=float(params.get("clahe_clip", 2.0)),
        clahe_tile=int(params.get("clahe_tile", 8)),
        order=params.get("order", "gamma_first"),
        channel=params.get("channel", "lab"),
    )
    return PipelineConfig(
        filter=filter_params,
        conf=float(params.get("conf", BASELINE_CONF)),
        imgsz=int(params.get("imgsz", BASELINE_IMGSZ)),
        enhance_crops=bool(params.get("enhance_crops", False))
        and not filter_params.is_identity,
    )


def suggest_config(trial, imgsz_choices: list[int]) -> PipelineConfig:
    """Search space. Conditional parameters are only sampled when they apply."""
    params: dict[str, Any] = {}
    params["gamma_mode"] = trial.suggest_categorical(
        "gamma_mode", ["off", "fixed", "adaptive"])
    if params["gamma_mode"] == "fixed":
        params["gamma"] = trial.suggest_float("gamma", *GAMMA_RANGE)
    elif params["gamma_mode"] == "adaptive":
        params["target_lightness"] = trial.suggest_float(
            "target_lightness", *TARGET_LIGHTNESS_RANGE)

    params["clahe"] = trial.suggest_categorical("clahe", [False, True])
    if params["clahe"]:
        params["clahe_clip"] = trial.suggest_float(
            "clahe_clip", *CLAHE_CLIP_RANGE, log=True)
        params["clahe_tile"] = trial.suggest_int("clahe_tile", *CLAHE_TILE_RANGE)

    filter_on = params["gamma_mode"] != "off" or params["clahe"]
    if params["gamma_mode"] != "off" and params["clahe"]:
        params["order"] = trial.suggest_categorical(
            "order", ["gamma_first", "clahe_first"])
    if filter_on:
        params["channel"] = trial.suggest_categorical("channel", ["lab", "hsv"])
        params["enhance_crops"] = trial.suggest_categorical(
            "enhance_crops", [False, True])

    params["conf"] = trial.suggest_float("conf", *CONF_RANGE)
    params["imgsz"] = trial.suggest_categorical("imgsz", imgsz_choices)
    return config_from_params(params)


def gate_filter(config: PipelineConfig, lightness: float,
                threshold: float | None) -> PipelineConfig:
    """Deployment behavior: only filter images darker than the threshold."""
    if threshold is None or lightness < threshold:
        return config
    return config.without_filter()


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruthPlate:
    box: Box
    text: str   # normalized; "" when not annotated

    @property
    def readable(self) -> bool:
        return PLATE_PATTERN.fullmatch(self.text) is not None


@dataclass
class Sample:
    """One annotated image, kept as its original encoded file bytes.

    Full-resolution frames are ~30 MB each when decoded, so they are decoded
    in small batches on demand instead of being held in memory.
    """
    path: Path
    encoded: bytes
    width: int
    height: int
    lightness: float
    plates: list[GroundTruthPlate]

    def decode(self) -> np.ndarray:
        bgr = cv2.imdecode(np.frombuffer(self.encoded, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not decode image: {self.path}")
        return bgr


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_yolo_boxes(path: Path, width: int, height: int) -> list[Box]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = map(float, parts)
        boxes.append(((xc - bw / 2) * width, (yc - bh / 2) * height,
                      (xc + bw / 2) * width, (yc + bh / 2) * height))
    return boxes


def read_json_plates(path: Path) -> list[tuple[Box, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    plates = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type", "rectangle") != "rectangle":
            continue
        (x1, y1), (x2, y2) = shape["points"][:2]
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        plates.append((box, normalize_plate(shape.get("text") or "")))
    return plates


def load_ground_truth(image_path: Path, labels_dir: Path, text_dir: Path,
                      width: int, height: int) -> tuple[list[GroundTruthPlate], list[str]]:
    """Boxes from the YOLO label, text from the JSON box with the highest IoU."""
    warnings: list[str] = []
    json_path = text_dir / f"{image_path.stem}.json"
    label_path = labels_dir / f"{image_path.stem}.txt"
    json_plates = read_json_plates(json_path) if json_path.is_file() else []
    if not json_path.is_file():
        warnings.append(f"{image_path.name}: no text annotation ({json_path.name})")

    if not label_path.is_file():
        warnings.append(f"{image_path.name}: no YOLO label, using JSON boxes")
        return [GroundTruthPlate(box, text) for box, text in json_plates], warnings

    plates = []
    unused = list(range(len(json_plates)))
    for box in read_yolo_boxes(label_path, width, height):
        best = max(unused, key=lambda i: box_iou(box, json_plates[i][0]), default=None)
        if best is not None and box_iou(box, json_plates[best][0]) >= TEXT_PAIR_IOU:
            unused.remove(best)
            plates.append(GroundTruthPlate(box, json_plates[best][1]))
        else:
            warnings.append(f"{image_path.name}: YOLO box without matching text")
            plates.append(GroundTruthPlate(box, ""))
    for i in unused:
        warnings.append(f"{image_path.name}: JSON box '{json_plates[i][1]}' "
                        "has no YOLO box (ignored)")
    return plates, warnings


def load_samples(paths: list[Path], labels_dir: Path,
                 text_dir: Path) -> tuple[list[Sample], list[str]]:
    samples, warnings = [], []
    for path in paths:
        encoded = path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not decode image: {path}")
        height, width = bgr.shape[:2]
        plates, issues = load_ground_truth(path, labels_dir, text_dir, width, height)
        warnings.extend(issues)
        samples.append(Sample(path, encoded, width, height, mean_lightness(bgr), plates))
        del bgr
    return samples, warnings


# --------------------------------------------------------------------------
# Pipeline and scoring
# --------------------------------------------------------------------------

@dataclass
class Detection:
    box: Box
    conf: float
    text: str | None = None      # OCR text; None when below the threshold
    gt_index: int | None = None  # matched ground truth at the threshold
    ap_tp: bool = False          # true positive in the all-confidence matching


@dataclass
class ImageResult:
    sample: Sample
    config: PipelineConfig       # after gating
    detections: list[Detection]  # all >= DETECTION_FLOOR, highest confidence first
    plate_rows: list[dict[str, Any]]
    false_positives: int
    crops: list[dict[str, Any]] = field(default_factory=list)  # for the plate sheet

    @property
    def kept(self) -> list[Detection]:
        return [d for d in self.detections if d.text is not None]


def greedy_match(pred_boxes: list[Box], gt_boxes: list[Box]) -> list[tuple[int | None, float]]:
    """Match predictions (highest confidence first) one-to-one at IoU >= MATCH_IOU."""
    taken: set[int] = set()
    matches = []
    for box in pred_boxes:
        best, best_iou = None, MATCH_IOU
        for index, gt_box in enumerate(gt_boxes):
            if index in taken:
                continue
            iou = box_iou(box, gt_box)
            if iou >= best_iou:
                best, best_iou = index, iou
        if best is None:
            matches.append((None, 0.0))
        else:
            taken.add(best)
            matches.append((best, best_iou))
    return matches


class Pipeline:
    """Detection + crop + OCR, matching the BentoML services, run in-process."""

    def __init__(self, device: str | int):
        import bentoml
        from fast_plate_ocr import LicensePlateRecognizer
        from ultralytics import YOLO

        detector_ref = bentoml.models.get("license_plate_detector:latest")
        ocr_ref = bentoml.models.get("license_plate_ocr:latest")
        print(f"Detector: {detector_ref.tag}")
        print(f"OCR:      {ocr_ref.tag}")

        self.device = device
        self.detector = YOLO(detector_ref.path_of("best.pt"))
        self.recognizer = LicensePlateRecognizer(
            onnx_model_path=ocr_ref.path_of("model.onnx"),
            plate_config_path=ocr_ref.path_of("plate_config.yaml"),
            device="cpu",
        )
        self.ocr_channels = self.recognizer.config.num_channels

    def detect(self, images_bgr: list[np.ndarray], conf: float,
               imgsz: int) -> list[list[tuple[list[float], float]]]:
        """Return, per image, a list of (xyxy box, confidence)."""
        results = self.detector.predict(
            source=images_bgr, imgsz=imgsz, conf=conf, iou=IOU_THRESHOLD,
            device=self.device, verbose=False,
        )
        outputs = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                outputs.append([])
                continue
            outputs.append(list(zip(boxes.xyxy.cpu().tolist(),
                                    boxes.conf.cpu().tolist())))
        return outputs

    def read_plate(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr.size == 0:
            return ""
        if self.ocr_channels == 1:
            array = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        else:
            array = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        return normalize_plate(self.recognizer.run_one(array).plate)

    def read_box(self, image: np.ndarray, box) -> str:
        height, width = image.shape[:2]
        try:
            x1, y1, x2, y2 = clamp_bbox(box, width, height)
        except ValueError:
            return ""
        return self.read_plate(image[y1:y2, x1:x2])

    def run(self, samples: list[Sample], config: PipelineConfig,
            gate: float | None = None, visuals: "Visuals | None" = None) -> list[ImageResult]:
        """Run the pipeline on samples, score them, and optionally save visuals."""
        results = []
        # Small batches keep only a few decoded full-resolution frames in memory.
        for start in range(0, len(samples), BATCH_SIZE):
            batch = samples[start:start + BATCH_SIZE]
            originals = [sample.decode() for sample in batch]
            configs = [gate_filter(config, s.lightness, gate) for s in batch]
            filtered = [apply_filter(image, cfg.filter)
                        for image, cfg in zip(originals, configs)]
            predictions = self.detect(filtered, DETECTION_FLOOR, config.imgsz)
            for sample, original, image, preds, cfg in zip(
                    batch, originals, filtered, predictions, configs):
                result = self._score(sample, original, image, preds, cfg,
                                     keep_crops=visuals is not None)
                if visuals is not None:
                    visuals.save_image(result, image)
                results.append(result)
            del originals, filtered
        if visuals is not None:
            visuals.save_plate_sheet(results)
        return results

    def _score(self, sample: Sample, original: np.ndarray, filtered: np.ndarray,
               predictions, config: PipelineConfig, keep_crops: bool) -> ImageResult:
        crop_source = filtered if config.enhance_crops else original
        detections = [Detection(tuple(box), float(conf))
                      for box, conf in sorted(predictions, key=lambda p: -p[1])]
        gt_boxes = [plate.box for plate in sample.plates]

        # AP: match every detection above the floor.
        for det, (gt_index, _) in zip(detections,
                                      greedy_match([d.box for d in detections], gt_boxes)):
            det.ap_tp = gt_index is not None

        # Operating point: detections at the confidence threshold get OCR.
        kept = [d for d in detections if d.conf >= config.conf]
        for det in kept:
            det.text = self.read_box(crop_source, det.box)
        matched: dict[int, tuple[Detection, float]] = {}
        for det, (gt_index, iou) in zip(kept, greedy_match([d.box for d in kept], gt_boxes)):
            det.gt_index = gt_index
            if gt_index is not None:
                matched[gt_index] = (det, iou)

        rows, crops = [], []
        for index, plate in enumerate(sample.plates):
            det, iou = matched.get(index, (None, 0.0))
            e2e_text = det.text if det is not None else ""
            gt_crop_text = self.read_box(crop_source, plate.box) if plate.readable else ""
            rows.append({
                "filename": sample.path.name,
                "plate_index": index,
                "gt_text": plate.text,
                "readable": plate.readable,
                "detected": det is not None,
                "detection_confidence": round(det.conf, 4) if det else None,
                "iou": round(iou, 3),
                "e2e_text": e2e_text,
                "e2e_correct": plate.readable and det is not None and e2e_text == plate.text,
                "e2e_edit_distance": edit_distance(plate.text, e2e_text)
                if plate.readable else None,
                "ocr_gt_crop_text": gt_crop_text,
                "ocr_gt_crop_correct": plate.readable and gt_crop_text == plate.text,
                "ocr_gt_crop_edit_distance": edit_distance(plate.text, gt_crop_text)
                if plate.readable else None,
                "lightness": round(sample.lightness, 4),
                "filter_applied": not config.filter.is_identity,
            })
            if keep_crops:
                try:
                    x1, y1, x2, y2 = clamp_bbox(plate.box, sample.width, sample.height)
                except ValueError:
                    continue
                crops.append({
                    "row": rows[-1],
                    "original": original[y1:y2, x1:x2].copy(),
                    "filtered": filtered[y1:y2, x1:x2].copy(),
                    "ocr_source": "filtered" if config.enhance_crops else "original",
                })

        false_positives = sum(d.gt_index is None for d in kept)
        return ImageResult(sample, config, detections, rows, false_positives, crops)


def average_precision(scored: list[tuple[float, bool]], total_gt: int) -> float:
    """All-point interpolated AP over (confidence, is_true_positive) pairs."""
    if total_gt == 0:
        return 0.0
    scored = sorted(scored, key=lambda item: -item[0])
    tp = np.cumsum([bool(t) for _, t in scored], dtype=float)
    fp = np.cumsum([not t for _, t in scored], dtype=float)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def aggregate(results: list[ImageResult]) -> dict[str, float]:
    rows = [row for result in results for row in result.plate_rows]
    readable = [row for row in rows if row["readable"]]
    chars = sum(len(row["gt_text"]) for row in readable)

    tp = sum(row["detected"] for row in rows)
    fp = sum(result.false_positives for result in results)
    fn = len(rows) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(rows) if rows else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap50 = average_precision(
        [(d.conf, d.ap_tp) for result in results for d in result.detections], len(rows))

    e2e_correct = sum(row["e2e_correct"] for row in readable)
    ocr_correct = sum(row["ocr_gt_crop_correct"] for row in readable)
    with_readable = [r for r in results if any(row["readable"] for row in r.plate_rows)]
    all_correct = sum(all(row["e2e_correct"] for row in r.plate_rows if row["readable"])
                      for r in with_readable)
    return {
        "images": len(results),
        "gt_plates": len(rows),
        "readable_plates": len(readable),
        "det_tp": tp, "det_fp": fp, "det_fn": fn,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "det_ap50": ap50,
        "ocr_correct": ocr_correct,
        "ocr_accuracy": ocr_correct / len(readable) if readable else 0.0,
        "ocr_cer": sum(r["ocr_gt_crop_edit_distance"] for r in readable) / chars
        if chars else 0.0,
        "e2e_correct": e2e_correct,
        "e2e_accuracy": e2e_correct / len(readable) if readable else 0.0,
        "e2e_cer": sum(r["e2e_edit_distance"] for r in readable) / chars if chars else 0.0,
        "images_all_correct": all_correct,
        "images_with_readable": len(with_readable),
    }


def objective_score(metrics: dict[str, float]) -> float:
    """End-to-end plate accuracy, with small tie-breakers so near-misses still count."""
    return (metrics["e2e_accuracy"]
            + 0.01 * metrics["det_f1"]
            + 0.01 * (1.0 - min(1.0, metrics["e2e_cer"])))


def format_metrics(m: dict[str, float]) -> str:
    return (f"e2e {m['e2e_correct']}/{m['readable_plates']} ({m['e2e_accuracy']:.1%}), "
            f"CER {m['e2e_cer']:.1%} | OCR on GT crops {m['ocr_correct']}/"
            f"{m['readable_plates']} ({m['ocr_accuracy']:.1%}) | det P "
            f"{m['det_precision']:.3f} R {m['det_recall']:.3f} F1 {m['det_f1']:.3f} "
            f"AP50 {m['det_ap50']:.3f}")


# --------------------------------------------------------------------------
# Visual outputs
# --------------------------------------------------------------------------

GREEN, YELLOW, RED, MAGENTA = (60, 200, 60), (0, 215, 255), (40, 40, 230), (220, 60, 220)
WHITE, GRAY, BLACK = (255, 255, 255), (150, 150, 150), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def write_jpeg(path: Path, image: np.ndarray, quality: int = 88) -> None:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def put_label(image: np.ndarray, text: str, x: int, y: int, color,
              scale: float = 0.5, above: bool = True) -> None:
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x = max(0, min(image.shape[1] - tw - 4, x))
    top = y - th - baseline - 4 if above else y
    top = max(0, min(image.shape[0] - th - baseline - 4, top))
    cv2.rectangle(image, (x, top), (x + tw + 4, top + th + baseline + 4), BLACK, -1)
    cv2.putText(image, text, (x + 2, top + th + 2), FONT, scale, color, 1, cv2.LINE_AA)


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.size == 0:
        return np.zeros((height, width, 3), np.uint8)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


class Visuals:
    """Writes filtered frames, annotated frames and a plate sheet into one folder."""

    def __init__(self, directory: Path, width: int, title: str):
        self.directory = directory
        self.width = width
        self.title = title

    def save_image(self, result: ImageResult, filtered: np.ndarray) -> None:
        sample = result.sample
        scale = min(1.0, self.width / sample.width)
        size = (round(sample.width * scale), round(sample.height * scale))
        small = cv2.resize(filtered, size, interpolation=cv2.INTER_AREA)
        stem = sample.path.stem
        write_jpeg(self.directory / "filtered" / f"{stem}.jpg", small)

        canvas = small.copy()
        s = lambda box: tuple(int(round(v * scale)) for v in box)  # noqa: E731
        matched = {d.gt_index for d in result.kept if d.gt_index is not None}

        for index, plate in enumerate(sample.plates):
            x1, y1, x2, y2 = s(plate.box)
            color = GREEN if index in matched else MAGENTA
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            label = f"GT {plate.text or '?'}" + ("" if index in matched else "  MISSED")
            put_label(canvas, label, x1, y2 + 2, color if plate.text else GRAY,
                      scale=0.45, above=False)

        for det in result.kept:
            x1, y1, x2, y2 = s(det.box)
            if det.gt_index is None:
                color = RED
            elif det.text == sample.plates[det.gt_index].text:
                color = GREEN
            else:
                color = YELLOW
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            put_label(canvas, f"{det.text or '<empty>'} {det.conf:.2f}", x1, y1 - 2, color)

        rows = result.plate_rows
        readable = [r for r in rows if r["readable"]]
        header = [
            self.title,
            f"{result.config.describe()} | filter "
            f"{'ON' if not result.config.filter.is_identity else 'off'} "
            f"(lightness {sample.lightness:.3f})",
            f"plates read {sum(r['e2e_correct'] for r in readable)}/{len(readable)} | "
            f"detected {sum(r['detected'] for r in rows)}/{len(rows)} | "
            f"false positives {result.false_positives}   "
            "[green = correct, yellow = misread, red = false positive, magenta = missed]",
        ]
        bar = np.zeros((22 * len(header) + 8, canvas.shape[1], 3), np.uint8)
        for i, line in enumerate(header):
            cv2.putText(bar, line, (8, 22 * (i + 1)), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        write_jpeg(self.directory / "annotated" / f"{stem}.jpg", np.vstack([bar, canvas]))

    def save_plate_sheet(self, results: list[ImageResult]) -> None:
        crop_w, crop_h, text_w, pad = 240, 80, 620, 8
        row_h = crop_h + pad
        width = 2 * crop_w + text_w + 4 * pad
        header = np.zeros((56, width, 3), np.uint8)
        cv2.putText(header, self.title, (pad, 20), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(header, "original crop", (pad, 46), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        cv2.putText(header, "filtered crop", (crop_w + 2 * pad, 46), FONT, 0.5, GRAY,
                    1, cv2.LINE_AA)
        cv2.putText(header, "ground truth | OCR on GT crop | pipeline", (2 * crop_w + 3 * pad, 46),
                    FONT, 0.5, GRAY, 1, cv2.LINE_AA)

        rows = [header]
        for result in results:
            for crop in result.crops:
                row = crop["row"]
                panel = np.full((row_h, width, 3), 24, np.uint8)
                panel[pad // 2:pad // 2 + crop_h, pad:pad + crop_w] = fit(
                    crop["original"], crop_w, crop_h)
                x0 = crop_w + 2 * pad
                panel[pad // 2:pad // 2 + crop_h, x0:x0 + crop_w] = fit(
                    crop["filtered"], crop_w, crop_h)
                # Mark the crop the OCR actually read.
                ocr_x = pad if crop["ocr_source"] == "original" else x0
                cv2.rectangle(panel, (ocr_x - 2, pad // 2 - 2),
                              (ocr_x + crop_w + 1, pad // 2 + crop_h + 1), YELLOW, 1)

                tx = 2 * crop_w + 3 * pad
                gt = row["gt_text"] or "<no text>"
                cv2.putText(panel, f"{gt}   {row['filename'][:30]}", (tx, 24), FONT, 0.55,
                            WHITE if row["readable"] else GRAY, 1, cv2.LINE_AA)
                if row["readable"]:
                    ocr_ok = row["ocr_gt_crop_correct"]
                    cv2.putText(panel, f"OCR on GT crop: {row['ocr_gt_crop_text'] or '<empty>'}",
                                (tx, 50), FONT, 0.55, GREEN if ocr_ok else RED, 1, cv2.LINE_AA)
                    e2e = row["e2e_text"] if row["detected"] else "<not detected>"
                    cv2.putText(panel, f"pipeline: {e2e or '<empty>'}", (tx, 76), FONT, 0.55,
                                GREEN if row["e2e_correct"] else RED, 1, cv2.LINE_AA)
                else:
                    cv2.putText(panel, "not scored for OCR (incomplete or no text)",
                                (tx, 50), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
                rows.append(panel)
        write_jpeg(self.directory / "plates.jpg", np.vstack(rows), quality=92)


def write_run_outputs(directory: Path, config: PipelineConfig, metrics: dict[str, float],
                      results: list[ImageResult], extra: dict[str, Any] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(
        {"description": config.describe(), "config": config.to_dict(), **(extra or {})},
        indent=2, default=str), encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(directory / "plates.csv",
              [row for result in results for row in result.plate_rows])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, samples: list[Sample], config: PipelineConfig,
             directory: Path | None, title: str, visual_width: int,
             gate: float | None = None, extra: dict[str, Any] | None = None):
    """Run one configuration; write metrics, plate CSV and (optionally) visuals."""
    visuals = Visuals(directory, visual_width, title) if directory is not None else None
    results = pipeline.run(samples, config, gate=gate, visuals=visuals)
    metrics = aggregate(results)
    if directory is not None:
        write_run_outputs(directory, config, metrics, results, extra)
    return results, metrics


def run_study(pipeline: Pipeline, samples: list[Sample], n_trials: int, seed: int,
              imgsz_choices: list[int], label: str, study_dir: Path,
              save_visuals: bool, visual_width: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=label,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=f"sqlite:///{(study_dir / 'study.db').as_posix()}",
    )
    study.enqueue_trial(BASELINE_TRIAL)
    study.enqueue_trial(PREVIOUS_FILTER_TRIAL)

    def objective(trial) -> float:
        config = suggest_config(trial, imgsz_choices)
        directory = study_dir / f"trial_{trial.number:03d}"
        _, metrics = evaluate(
            pipeline, samples, config,
            directory if save_visuals else None,
            f"{label} trial {trial.number}", visual_width,
            extra={"trial": trial.number, "params": trial.params})
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return objective_score(metrics)

    start = time.perf_counter()

    def progress(study, trial) -> None:
        done = trial.number + 1
        if done % 10 == 0 or done == n_trials:
            best = study.best_trial.user_attrs
            print(f"  [{label}] {done}/{n_trials} trials, best e2e "
                  f"{best['e2e_correct']}/{best['readable_plates']} "
                  f"({best['e2e_accuracy']:.1%}) ({time.perf_counter() - start:.0f} s)",
                  flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    return study


def k_folds(count: int, folds: int, seed: int) -> list[list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return [indices[fold::folds] for fold in range(folds)]


# --------------------------------------------------------------------------
# Detection test split (daylight regression check)
# --------------------------------------------------------------------------

def detection_split_check(pipeline: Pipeline, configs: dict[str, PipelineConfig],
                          threshold: float) -> dict[str, Any]:
    """Precision/recall/F1 at IoU 0.5 on the detection test split, per config."""
    image_dir = PROJECT_ROOT / "datasets/detection/raw/images/test"
    label_dir = PROJECT_ROOT / "datasets/detection/raw/labels/test"
    if not image_dir.is_dir():
        return {"skipped": f"not found: {image_dir}"}

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    totals = {name: {"gt": 0, "pred": 0, "matched": 0} for name in configs}
    filtered_images = 0

    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        height, width = bgr.shape[:2]
        label_path = label_dir / f"{path.stem}.txt"
        gt_boxes = read_yolo_boxes(label_path, width, height) if label_path.is_file() else []
        lightness = mean_lightness(bgr)
        filtered_images += lightness < threshold

        for name, config in configs.items():
            gated = gate_filter(config, lightness, threshold)
            predictions = pipeline.detect([apply_filter(bgr, gated.filter)],
                                          gated.conf, gated.imgsz)[0]
            predictions.sort(key=lambda item: -item[1])
            matches = greedy_match([box for box, _ in predictions], gt_boxes)
            totals[name]["gt"] += len(gt_boxes)
            totals[name]["pred"] += len(predictions)
            totals[name]["matched"] += sum(g is not None for g, _ in matches)

    report: dict[str, Any] = {"images": len(paths), "images_darker_than_gate": filtered_images}
    for name, total in totals.items():
        recall = total["matched"] / total["gt"] if total["gt"] else 0.0
        precision = total["matched"] / total["pred"] if total["pred"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {**total, "recall": recall, "precision": precision, "f1": f1}
    return report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and tune gamma correction + CLAHE preprocessing "
                    "for the license plate pipeline.")
    parser.add_argument("--image-dir", type=Path,
                        default=Path("datasets/absolute_test/anotated_yolo/images"))
    parser.add_argument("--labels-dir", type=Path,
                        default=Path("datasets/absolute_test/anotated_yolo/labels"),
                        help="YOLO labels with the plate boxes.")
    parser.add_argument("--text-dir", type=Path,
                        default=Path("datasets/absolute_test/cropped_images"),
                        help="JSON annotations with the plate text ('text' field).")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        help="Filenames kept out of tuning (daylight images); "
                             "they are used for the daylight check instead.")
    parser.add_argument("--trials", type=int, default=60,
                        help="Optuna trials per search (default: 60).")
    parser.add_argument("--folds", type=int, default=5,
                        help="Outer cross-validation folds (default: 5).")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280, 1600],
                        help="Detector inference resolutions to search.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="Detector device, e.g. 0 or cpu (default: GPU if available).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: runs/pipeline/tuning/<timestamp>.")
    parser.add_argument("--visuals", choices=["all", "final", "none"], default="all",
                        help="Save filtered/annotated images and plate sheets for every "
                             "trial of every search (all), only the final search and "
                             "held-out/daylight evaluations (final), or none.")
    parser.add_argument("--visual-width", type=int, default=1600,
                        help="Width of saved full-frame images (default: 1600).")
    parser.add_argument("--evaluate", metavar="CONFIG", default=None,
                        help="Only evaluate 'baseline' or a best_params.json on all "
                             "images (no search).")
    parser.add_argument("--skip-nested-cv", action="store_true",
                        help="Only run the final search (no unbiased estimate).")
    parser.add_argument("--skip-day-check", action="store_true",
                        help="Skip the detection test split daylight check.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def print_section(title: str) -> None:
    print(f"\n{title}\n" + "-" * 76)


def run_evaluate_only(args, pipeline: Pipeline, night: list[Sample], day: list[Sample],
                      threshold: float, output_dir: Path) -> dict[str, Any]:
    if args.evaluate == "baseline":
        config, gate = BASELINE, threshold
    else:
        data = json.loads(resolve(Path(args.evaluate)).read_text(encoding="utf-8"))
        config = config_from_params(data["params"])
        gate = data.get("darkness_threshold", threshold)
    print_section(f"EVALUATE: {config.describe()} (darkness gate {gate})")
    directory = output_dir / "evaluate" if args.visuals != "none" else None
    results, metrics = evaluate(pipeline, night + day, config, directory,
                                f"evaluate {args.evaluate}", args.visual_width, gate=gate)
    night_ids = {id(s) for s in night}
    by_group = {
        "all": metrics,
        "night": aggregate([r for r in results if id(r.sample) in night_ids]),
        "day": aggregate([r for r in results if id(r.sample) not in night_ids]),
    }
    for name, m in by_group.items():
        print(f"  {name:6s} {format_metrics(m)}")
    write_csv(output_dir / "plates.csv",
              [row for result in results for row in result.plate_rows])
    return {"evaluate": args.evaluate, "config": config.to_dict(),
            "darkness_threshold": gate, "metrics": by_group}


def main() -> int:
    args = parse_args()
    import torch

    device = args.device if args.device is not None else (
        0 if torch.cuda.is_available() else "cpu")
    image_dir = resolve(args.image_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "runs/pipeline/tuning" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(p for p in image_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    excluded = set(args.exclude)
    night_paths = [p for p in all_paths if p.name not in excluded]
    day_paths = [p for p in all_paths if p.name in excluded]

    print("=" * 76)
    print("LOW-LIGHT PREPROCESSING — EVALUATION AND HYPERPARAMETER SEARCH")
    print("=" * 76)
    print(f"Images: {image_dir}")
    print(f"Boxes:  {resolve(args.labels_dir)}")
    print(f"Text:   {resolve(args.text_dir)}")

    night, warnings = load_samples(night_paths, resolve(args.labels_dir),
                                   resolve(args.text_dir))
    day, day_warnings = load_samples(day_paths, resolve(args.labels_dir),
                                     resolve(args.text_dir))
    for warning in warnings + day_warnings:
        print(f"  WARNING: {warning}")

    def plate_counts(samples):
        plates = [p for s in samples for p in s.plates]
        return len(plates), sum(p.readable for p in plates)

    n_plates, n_readable = plate_counts(night)
    d_plates, d_readable = plate_counts(day)
    print(f"Night (tuning): {len(night)} images, {n_plates} plates "
          f"({n_readable} with complete text)")
    print(f"Day (check):    {len(day)} images, {d_plates} plates "
          f"({d_readable} with complete text)")
    unscored = [f"{s.path.stem.split('-')[0]}:'{p.text}'"
                for s in night + day for p in s.plates if not p.readable]
    if unscored:
        print(f"Boxes without complete text (detection only): {', '.join(unscored)}")

    night_max = max(s.lightness for s in night)
    day_min = min((s.lightness for s in day), default=night_max + 0.1)
    threshold = round((night_max + day_min) / 2, 3)
    print(f"Mean lightness — night: {min(s.lightness for s in night):.3f}"
          f"–{night_max:.3f}, day: "
          + (", ".join(f"{s.lightness:.3f}" for s in day) or "n/a")
          + f" -> darkness gate {threshold}")
    print(f"Device: {device} | visuals: {args.visuals} | output: {output_dir}")

    print("\nLoading models...")
    pipeline = Pipeline(device)

    summary: dict[str, Any] = {
        "night_images": len(night), "night_plates": n_plates, "night_readable": n_readable,
        "day_images": len(day), "day_plates": d_plates, "day_readable": d_readable,
        "darkness_threshold": threshold, "seed": args.seed,
        "search_space": {
            "gamma_mode": ["off", "fixed", "adaptive"], "gamma": GAMMA_RANGE,
            "target_lightness": TARGET_LIGHTNESS_RANGE, "clahe": [False, True],
            "clahe_clip": f"{CLAHE_CLIP_RANGE} log", "clahe_tile": CLAHE_TILE_RANGE,
            "order": ["gamma_first", "clahe_first"], "channel": ["lab", "hsv"],
            "enhance_crops": [False, True], "conf": CONF_RANGE, "imgsz": args.imgsz,
        },
    }

    if args.evaluate:
        summary.update(run_evaluate_only(args, pipeline, night, day, threshold, output_dir))
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nOutputs: {output_dir}")
        return 0

    summary["trials_per_search"] = args.trials
    trial_visuals = args.visuals == "all"
    eval_visuals = args.visuals != "none"

    # ---- Nested cross-validation -------------------------------------------
    if not args.skip_nested_cv:
        print_section(f"NESTED CROSS-VALIDATION ({args.folds} folds, split by image)")
        held_out: dict[str, list[ImageResult]] = {
            "baseline": [], "tuned": [], "tuned_without_filter": []}
        per_plate: list[dict[str, Any]] = []
        fold_reports = []
        for fold, test_idx in enumerate(k_folds(len(night), args.folds, args.seed)):
            fold_dir = output_dir / "nested_cv" / f"fold{fold}"
            train = [night[i] for i in range(len(night)) if i not in test_idx]
            test = [night[i] for i in test_idx]
            study = run_study(pipeline, train, args.trials, args.seed + fold,
                              args.imgsz, f"fold{fold}", fold_dir,
                              trial_visuals, args.visual_width)
            tuned = config_from_params(study.best_params)
            for name, config in (("baseline", BASELINE), ("tuned", tuned),
                                 ("tuned_without_filter", tuned.without_filter())):
                results, _ = evaluate(
                    pipeline, test, config,
                    fold_dir / f"heldout_{name}" if eval_visuals else None,
                    f"fold{fold} held-out {name}", args.visual_width)
                held_out[name].extend(results)
                per_plate.extend({"fold": fold, "config": name, **row}
                                 for r in results for row in r.plate_rows)
            best_attrs = study.best_trial.user_attrs
            fold_reports.append({
                "fold": fold,
                "test_images": [s.path.name for s in test],
                "best_params": study.best_params,
                "train_metrics": best_attrs,
            })
            print(f"  fold {fold}: train e2e {best_attrs['e2e_correct']}/"
                  f"{best_attrs['readable_plates']}, {tuned.describe()}", flush=True)

        write_csv(output_dir / "nested_cv_per_plate.csv", per_plate)
        held_metrics = {name: aggregate(results) for name, results in held_out.items()}
        summary["nested_cv"] = {"held_out": held_metrics, "folds": fold_reports}
        print("\nHeld-out results (each night image scored once, unseen by its search):")
        for name, metrics in held_metrics.items():
            print(f"  {name:21s} {format_metrics(metrics)}")

    # ---- Final search on all night images ----------------------------------
    print_section("FINAL SEARCH (all night images)")
    study = run_study(pipeline, night, args.trials, args.seed, args.imgsz, "final",
                      output_dir / "final", eval_visuals, args.visual_width)
    best = config_from_params(study.best_params)
    best_attrs = study.best_trial.user_attrs

    importances: dict[str, float] = {}
    try:
        import warnings

        import optuna
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importances = {k: float(v) for k, v in
                           optuna.importance.get_param_importances(study).items()}
    except Exception as exc:  # importance needs enough completed trials
        print(f"  (parameter importance unavailable: {exc})")

    best_report = {
        "params": study.best_params,
        "trial": study.best_trial.number,
        "description": best.describe(),
        "config": best.to_dict(),
        "darkness_threshold": threshold,
        "tuning_metrics": best_attrs,
        "note": "Tuning metrics are optimistic; use the nested CV held-out "
                "result as the performance estimate.",
    }
    (output_dir / "best_params.json").write_text(
        json.dumps(best_report, indent=2, default=str), encoding="utf-8")
    summary["final"] = {**best_report, "param_importances": importances}
    print(f"  best (trial {study.best_trial.number}): {best.describe()}")
    print(f"  tuning metrics (optimistic): {format_metrics(best_attrs)}")
    if importances:
        print("  parameter importance: " + ", ".join(
            f"{k} {v:.2f}" for k, v in importances.items()))

    # ---- Daylight checks ----------------------------------------------------
    if day:
        print_section("DAYLIGHT IMAGES (gated by darkness threshold)")
        day_summary = {}
        for name, config in (("baseline", BASELINE), ("tuned", best)):
            results, metrics = evaluate(
                pipeline, day, config,
                output_dir / "daylight" / name if eval_visuals else None,
                f"daylight {name}", args.visual_width, gate=threshold)
            day_summary[name] = metrics
            print(f"  {name:8s} {format_metrics(metrics)}")
        summary["daylight_images"] = day_summary

    if not args.skip_day_check:
        print_section("DETECTION TEST SPLIT (daylight regression check, IoU 0.5)")
        check = detection_split_check(pipeline, {
            "baseline": BASELINE, "tuned": best,
            "tuned_without_filter": best.without_filter()}, threshold)
        summary["detection_test_split"] = check
        if "skipped" in check:
            print(f"  skipped: {check['skipped']}")
        else:
            print(f"  images: {check['images']}, darker than gate (filtered): "
                  f"{check['images_darker_than_gate']}")
            for name in ("baseline", "tuned", "tuned_without_filter"):
                c = check[name]
                print(f"  {name:21s} recall {c['recall']:.4f}, precision "
                      f"{c['precision']:.4f}, F1 {c['f1']:.4f}")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nOutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
