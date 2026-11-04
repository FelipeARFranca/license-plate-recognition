
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "detection" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_files(folder: Path, extensions: set[str]) -> list[Path]:
    return [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]


def inspect_directories() -> None:
    print("\n=== DIRECTORY STRUCTURE ===")

    directories = [DATASET_ROOT]
    directories.extend(
        sorted(path for path in DATASET_ROOT.rglob("*") if path.is_dir())
    )

    for directory in directories:
        images = get_files(directory, IMAGE_EXTENSIONS)
        labels = get_files(directory, {".txt"})

        if images or labels:
            relative_path = directory.relative_to(PROJECT_ROOT)
            print(
                f"{relative_path}: "
                f"{len(images)} images, {len(labels)} text files"
            )


def inspect_images() -> None:
    images = sorted(
        path
        for path in DATASET_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("\n=== IMAGE SUMMARY ===")
    print(f"Total images: {len(images)}")

    if not images:
        return

    sample = images[0]
    image = cv2.imread(str(sample))

    print(f"Sample image: {sample.relative_to(PROJECT_ROOT)}")

    if image is not None:
        height, width = image.shape[:2]
        print(f"Image dimensions: {width}x{height}")
    else:
        print("WARNING: Could not read the sample image.")


def inspect_labels() -> None:
    labels = sorted(DATASET_ROOT.rglob("*.txt"))

    print("\n=== ANNOTATION SUMMARY ===")
    print(f"Total TXT files: {len(labels)}")

    for label_file in labels[:3]:
        print(f"\nSample: {label_file.relative_to(PROJECT_ROOT)}")

        try:
            lines = label_file.read_text(
                encoding="utf-8",
                errors="replace"
            ).splitlines()

            for line in lines[:5]:
                print(f"  {line}  [columns: {len(line.split())}]")

            if not lines:
                print("  Empty annotation file.")

        except OSError as error:
            print(f"  Could not read file: {error}")


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    print(f"Dataset root: {DATASET_ROOT}")

    inspect_directories()
    inspect_images()
    inspect_labels()


if __name__ == "__main__":
    main()
