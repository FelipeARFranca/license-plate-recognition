from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "datasets/detection/raw/images/train"
LABEL_DIR = PROJECT_ROOT / "datasets/detection/raw/labels/train"

OUTPUT_DIR = PROJECT_ROOT / "runs/detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_samples():
    positive = []
    negative = []

    for image_path in IMAGE_DIR.iterdir():
        if image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            continue

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        if label_path.read_text(encoding="utf-8").strip():
            positive.append(image_path)
        else:
            negative.append(image_path)

    rng = random.Random(42)

    samples = rng.sample(positive, min(10, len(positive)))
    samples += rng.sample(negative, min(2, len(negative)))

    return samples


def preview():
    samples = get_samples()

    if not samples:
        raise RuntimeError("No annotated images found.")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for ax, image_path in zip(axes, samples):
        image = cv2.imread(str(image_path))

        if image is None:
            ax.set_title(f"{image_path.name} - unreadable")
            ax.axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        box_count = 0

        for line in lines:
            if not line.strip():
                continue

            class_id, xc, yc, bw, bh = map(float, line.split())

            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height

            # Clip only for visualization. Original labels stay unchanged.
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )

            box_count += 1

        ax.imshow(image)
        ax.set_title(f"{image_path.name}\n{box_count} plates")
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.tight_layout()

    output_path = OUTPUT_DIR / "annotation_preview.png"
    fig.savefig(output_path, dpi=150)
    print(f"Preview saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    preview()
