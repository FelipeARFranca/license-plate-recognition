
import bentoml
import torch

from PIL.Image import Image
from ultralytics import YOLO


@bentoml.service(
    name="license_plate_detection",
    workers=1,
)
class LicensePlateDetection:

    # Reference to the model registered in BentoML.
    model_ref = bentoml.models.BentoModel(
        "license_plate_detector:latest"
    )

    def __init__(self):
        # Select GPU when available, otherwise use CPU.
        self.device = 0 if torch.cuda.is_available() else "cpu"

        print(f"Loading YOLO model on device: {self.device}")

        # Load the checkpoint from BentoML's Model Store.
        model_path = self.model_ref.path_of("best.pt")
        self.model = YOLO(model_path)

        print("YOLO model successfully loaded!")

    @bentoml.api
    def detect(
        self,
        image: Image,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> dict:

        # Validate inference parameters.
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        # Ensure the image has three color channels.
        image = image.convert("RGB")

        # Run YOLO inference.
        results = self.model.predict(
            source=image,
            imgsz=640,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        detections = []

        if boxes is not None:
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()

            for xyxy, confidence, class_id in zip(
                coordinates,
                confidences,
                class_ids,
            ):
                class_id = int(class_id)

                if isinstance(result.names, list):
                    class_name = result.names[class_id]
                else:
                    class_name = result.names.get(
                        class_id,
                        str(class_id),
                    )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 6),
                    "bbox_xyxy": [
                        round(float(value), 2)
                        for value in xyxy
                    ],
                })

        return {
            "image": {
                "width": image.width,
                "height": image.height,
            },
            "detection_count": len(detections),
            "coordinate_format": "xyxy_pixels",
            "detections": detections,
        }
