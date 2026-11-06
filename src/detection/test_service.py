
from pathlib import Path

import bentoml


def main():
    test_images = Path(
        "datasets/detection/raw/images/test"
    )

    supported_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    images = sorted(
        path
        for path in test_images.iterdir()
        if path.suffix.lower() in supported_extensions
    )

    if not images:
        raise FileNotFoundError(
            f"No test images found in {test_images}"
        )

    image_path = images[0]

    print(f"Sending image: {image_path}")

    client = bentoml.SyncHTTPClient(
        "http://localhost:3000"
    )

    try:
        response = client.detect(
            image=image_path,
            confidence_threshold=0.25,
            iou_threshold=0.7,
        )

        print("\n===== INFERENCE RESULT =====")
        print(f"Image: {image_path.name}")
        print(f"Detections: {response['detection_count']}")

        for index, detection in enumerate(
            response["detections"],
            start=1,
        ):
            print(f"\nDetection {index}")
            print(f"Class: {detection['class_name']}")
            print(f"Confidence: {detection['confidence']}")
            print(f"Bounding box: {detection['bbox_xyxy']}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
