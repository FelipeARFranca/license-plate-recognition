from pathlib import Path
from math import isfinite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "detection" / "raw"

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOLERANCE = 1e-5


def validate_split(split: str) -> bool:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"[ERROR] Missing directories for {split}")
        return False

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(label_dir.glob("*.txt"))

    images_by_stem = {p.stem: p for p in images}
    labels_by_stem = {p.stem: p for p in labels}

    errors = []
    boxes = 0
    empty_labels = 0

    if len(images_by_stem) != len(images):
        errors.append("Duplicate image filenames (same stem).")

    if len(labels_by_stem) != len(labels):
        errors.append("Duplicate annotation filenames (same stem).")

    missing = set(images_by_stem) - set(labels_by_stem)
    extra = set(labels_by_stem) - set(images_by_stem)

    if missing:
        errors.append(f"{len(missing)} images without annotation files.")

    if extra:
        errors.append(f"{len(extra)} annotations without matching images.")

    boundary_warnings = []

    for stem in set(images_by_stem) & set(labels_by_stem):
        label_path = labels_by_stem[stem]

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            errors.append(f"{label_path.name}: {error}")
            continue

        valid_lines = [line for line in lines if line.strip()]

        if not valid_lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(valid_lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    f"Expected 5 values, got {len(parts)}."
                )
                continue

            try:
                class_value = float(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(
                    f"{label_path.name}:{line_number}: Non-numeric value."
                )
                continue

            if not isfinite(class_value) or not class_value.is_integer():
                errors.append(
                    f"{label_path.name}:{line_number}: Invalid class ID."
                )
                continue

            class_id = int(class_value)

            if class_id != 0:
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    f"Unexpected class ID {class_id}."
                )
                continue

            if not all(isfinite(value) for value in coords):
                errors.append(
                    f"{label_path.name}:{line_number}: Non-finite coordinates."
                )
                continue

            x_center, y_center, width, height = coords

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                errors.append(
                    f"{label_path.name}:{line_number}: "
                    "Coordinates outside valid normalized ranges."
                )
                continue

            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2

            overflow = max(
                0.0,
                -x_min,
                x_max - 1.0,
                -y_min,
                y_max - 1.0,
            )

            if overflow > TOLERANCE:
                boundary_warnings.append(
                    (label_path.name, line_number, overflow)
                )

            boxes += 1

    print(f"\n=== {split.upper()} ===")
    print(f"Images: {len(images)}")
    print(f"Annotation files: {len(labels)}")
    print(f"Bounding boxes: {boxes}")
    print(f"Empty annotations: {empty_labels}")
    print(f"Errors: {len(errors)}")

    for error in errors[:10]:
        print(f"[ERROR] {error}")

    if len(errors) > 10:
        print(f"... and {len(errors) - 10} additional errors.")

    print(f"Boundary warnings: {len(boundary_warnings)}")

    for filename, line_number, overflow in boundary_warnings[:10]:
        print(
            f"[WARNING] {filename}:{line_number}: "
            f"Maximum normalized boundary overflow: {overflow:.6f}"
        )

    if len(boundary_warnings) > 10:
        print(
            f"... and {len(boundary_warnings) - 10} additional warnings."
        )

    return not errors


def main():
    print(f"Dataset: {DATASET_ROOT}")

    results = [
        validate_split(split)
        for split in SPLITS
    ]

    if all(results):
        print("\n[SUCCESS] Dataset validation passed!")
    else:
        raise SystemExit("\n[FAILURE] Fix the dataset issues before training.")


if __name__ == "__main__":
    main()
