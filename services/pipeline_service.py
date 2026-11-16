from __future__ import annotations

import base64
import io
import math
import time
from typing import Any

import bentoml
from PIL import Image, ImageDraw, ImageFont

from services.detection_service import LicensePlateDetection
from services.ocr_service import LicensePlateOCR


def encode_png_base64(image: Image.Image) -> dict[str, str]:
    #Encode a Pillow image as a Base64 PNG payload.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "mime_type": "image/png",
        "encoding": "base64",
        "data": encoded,
    }


def clamp_bbox(
    bbox_xyxy: list[float] | tuple[float, ...],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    #Round and clamp an XYXY bounding box to valid image coordinates.
    if len(bbox_xyxy) != 4:
        raise ValueError("A bounding box must contain exactly four coordinates.")

    x1_raw, y1_raw, x2_raw, y2_raw = (float(value) for value in bbox_xyxy)
    if not all(math.isfinite(value) for value in (x1_raw, y1_raw, x2_raw, y2_raw)):
        raise ValueError("Bounding-box coordinates must be finite numbers.")

    # Floor the top-left and ceil the bottom-right to avoid cutting off edge pixels.
    x1 = max(0, min(image_width, math.floor(x1_raw)))
    y1 = max(0, min(image_height, math.floor(y1_raw)))
    x2 = max(0, min(image_width, math.ceil(x2_raw)))
    y2 = max(0, min(image_height, math.ceil(y2_raw)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Bounding box has no visible area after clipping: "
            f"{(x1, y1, x2, y2)}"
        )

    return x1, y1, x2, y2


def draw_detection(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    image_width: int,
    image_height: int,
) -> None:
    #Draw a colored rectangle and readable label for one detection.
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=3)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = 4

    label_left = x1
    label_top = max(0, y1 - text_height - (padding * 2))
    label_right = min(
        image_width,
        label_left + text_width + padding * 2,
    )
    label_bottom = min(
        image_height,
        label_top + text_height + padding * 2,
    )

    draw.rectangle(
        (label_left, label_top, label_right, label_bottom),
        fill=color,
    )
    draw.text(
        (label_left + padding, label_top + padding),
        label,
        fill=(0, 0, 0),
        font=font,
    )


@bentoml.service(name="license_plate_pipeline", workers=1)
class LicensePlatePipeline:
    #Coordinate detection, cropping, OCR, and visual result generation.

    detector = bentoml.depends(LicensePlateDetection)
    recognizer = bentoml.depends(LicensePlateOCR)

    @bentoml.api
    def recognize(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        return_ocr_confidence: bool = True,
    ) -> dict[str, Any]:
        
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a decoded PIL image.")

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        original = image.convert("RGB")
        image_width, image_height = original.size
        if image_width <= 0 or image_height <= 0:
            raise ValueError("The input image has invalid dimensions.")

        pipeline_start = time.perf_counter()

        # Detect all license plates in the original image.
        detection_start = time.perf_counter()
        detection_result = self.detector.detect(
            original,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        detection_ms = (time.perf_counter() - detection_start) * 1000.0

        raw_detections = detection_result.get("detections", [])
        if not isinstance(raw_detections, list):
            raise RuntimeError("Detection service returned an invalid detections field.")

        # Prepare an annotated copy. The original remains unchanged for cropping.
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        palette = [
            (0, 255, 0),
            (0, 200, 255),
            (255, 180, 0),
            (255, 80, 180),
            (180, 100, 255),
            (0, 220, 180),
        ]

        results: list[dict[str, Any]] = []
        crop_ms = 0.0
        crop_encoding_ms = 0.0
        ocr_ms = 0.0
        annotation_ms = 0.0
        successful_recognitions = 0

        # Crop each detection and send it to the OCR service.
        for index, detection in enumerate(raw_detections, start=1):
            item: dict[str, Any] = {
                "id": index,
                "class_id": detection.get("class_id"),
                "class_name": detection.get("class_name", "license_plate"),
                "detection_confidence": float(detection.get("confidence", 0.0)),
            }

            try:
                bbox = clamp_bbox(
                    detection["bbox_xyxy"],
                    image_width=image_width,
                    image_height=image_height,
                )
            except (KeyError, TypeError, ValueError) as exc:
                item.update({
                    "plate_text": None,
                    "ocr_status": "not_run",
                    "crop_error": str(exc),
                })
                results.append(item)
                continue

            x1, y1, x2, y2 = bbox
            item["bbox_xyxy"] = [x1, y1, x2, y2]

            crop_start = time.perf_counter()
            cropped_plate = original.crop((x1, y1, x2, y2))
            crop_ms += (time.perf_counter() - crop_start) * 1000.0

            # Keep the crop in the response even if the OCR request fails.
            encode_start = time.perf_counter()
            item["cropped_image"] = encode_png_base64(cropped_plate)
            crop_encoding_ms += (time.perf_counter() - encode_start) * 1000.0
            item["crop_dimensions"] = {
                "width": cropped_plate.width,
                "height": cropped_plate.height,
            }

            ocr_start = time.perf_counter()
            try:
                ocr_result = self.recognizer.recognize(
                    cropped_plate,
                    return_confidence=return_ocr_confidence,
                )
                item["plate_text"] = ocr_result.get("plate")
                item["ocr_status"] = "success"
                if "character_confidences" in ocr_result:
                    item["character_confidences"] = ocr_result[
                        "character_confidences"
                    ]
                if "region" in ocr_result:
                    item["region"] = ocr_result["region"]
                if "region_confidence" in ocr_result:
                    item["region_confidence"] = ocr_result["region_confidence"]
                successful_recognitions += 1
            except Exception as exc:
                item["plate_text"] = None
                item["ocr_status"] = "error"
                item["ocr_error"] = f"{type(exc).__name__}: {exc}"

            item["ocr_time_ms"] = round(
                (time.perf_counter() - ocr_start) * 1000.0, 3
            )
            ocr_ms += item["ocr_time_ms"]

            # Annotate the original image with box and recognized plate text.
            annotation_start = time.perf_counter()
            color = palette[(index - 1) % len(palette)]
            text = item["plate_text"] or "Unrecognized"
            label = f"{index}: {text} ({item['detection_confidence']:.2f})"
            draw_detection(
                draw,
                bbox,
                label,
                color,
                font,
                image_width=image_width,
                image_height=image_height,
            )
            annotation_ms += (time.perf_counter() - annotation_start) * 1000.0

            results.append(item)

        # Return the full annotated image, including when no plates were found.
        annotated_encode_start = time.perf_counter()
        annotated_payload = encode_png_base64(annotated)
        annotated_encoding_ms = (
            time.perf_counter() - annotated_encode_start
        ) * 1000.0
        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        return {
            "image": {
                "width": image_width,
                "height": image_height,
                "color_mode": "RGB",
            },
            "model": {
                "detector": "YOLO26n",
                "recognizer": "cct-s-v2-global-model",
            },
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
            },
            "detection_count": len(raw_detections),
            "successful_recognitions": successful_recognitions,
            "detections": results,
            "annotated_image": annotated_payload,
            "timings_ms": {
                "detection": round(detection_ms, 3),
                "cropping": round(crop_ms, 3),
                "crop_encoding": round(crop_encoding_ms, 3),
                "ocr": round(ocr_ms, 3),
                "annotation": round(annotation_ms, 3),
                "annotated_image_encoding": round(annotated_encoding_ms, 3),
                "total": round(total_ms, 3),
            },
        }
