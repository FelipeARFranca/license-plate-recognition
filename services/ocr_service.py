from __future__ import annotations

import numpy as np
import bentoml
from PIL.Image import Image
from fast_plate_ocr import LicensePlateRecognizer


@bentoml.service(name="license_plate_ocr", workers=1)
class LicensePlateOCR:

    model_ref = bentoml.models.BentoModel("license_plate_ocr:latest")

    def __init__(self) -> None:
        model_path = self.model_ref.path_of("model.onnx")
        config_path = self.model_ref.path_of("plate_config.yaml")

        # Use CPU ONNX Runtime consistently with the tested baseline.
        # The service loads local BentoML artifacts; it does not contact the Hub.
        self.recognizer = LicensePlateRecognizer(
            onnx_model_path=model_path,
            plate_config_path=config_path,
            device="cpu",
        )

    @bentoml.api
    def recognize(
        self,
        image: Image,
        return_confidence: bool = False,
    ) -> dict:
        
        if not isinstance(image, Image):
            raise TypeError("image must be a decoded PIL image.")

        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        if width <= 0 or height <= 0:
            raise ValueError("The uploaded image has invalid dimensions.")

        num_channels = self.recognizer.config.num_channels
        if num_channels == 1:
            image_array = np.asarray(rgb_image.convert("L"), dtype=np.uint8)
        elif num_channels == 3:
            image_array = np.asarray(rgb_image, dtype=np.uint8)
        else:
            raise RuntimeError(
                f"Unsupported model channel count: {num_channels}."
            )

        prediction = self.recognizer.run_one(
            image_array,
            return_confidence=return_confidence,
        )

        result = {
            "plate": str(prediction.plate),
            "image": {
                "width": int(width),
                "height": int(height),
            },
            "model": "cct-s-v2-global-model",
        }

        char_probs = getattr(prediction, "char_probs", None)
        if return_confidence and char_probs is not None:
            result["character_confidences"] = [
                float(value) for value in np.asarray(char_probs).reshape(-1)
            ]

        region = getattr(prediction, "region", None)
        if region is not None:
            result["region"] = str(region)

        region_prob = getattr(prediction, "region_prob", None)
        if region_prob is not None:
            result["region_confidence"] = float(region_prob)

        return result
