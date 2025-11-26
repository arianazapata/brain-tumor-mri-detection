# src/preprocess.py
import os
import cv2
import numpy as np
from pathlib import Path

SPLITS = ["train", "val", "test"]
CLASSES = ["tumor", "no_tumor"]
IMAGE_SIZE = (256, 256)  # (width, height)

DATA_DIR = Path("data")
PREPROCESSED_DIR = Path("preprocessed")
MASKS_DIR = Path("results") / "masks"
EDGES_DIR = Path("results") / "edges"


def ensure_dirs():
    for base in [PREPROCESSED_DIR, MASKS_DIR, EDGES_DIR]:
        for split in SPLITS:
            for cls in CLASSES:
                (base / split / cls).mkdir(parents=True, exist_ok=True)


def largest_component_mask(binary_img: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary image.
    If none found, return the original.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels <= 1:
        return binary_img

    # stats: [label, x, y, width, height, area], index 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    largest_mask = (labels == largest_label).astype(np.uint8) * 255
    return largest_mask


def process_single_image(in_path: Path, split: str, cls: str):
    # Load grayscale
    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: could not read {in_path}")
        return

    # Resize
    img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Gaussian blur
    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # Otsu threshold
    _, thresh_otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphology: opening then closing
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Keep largest connected component
    cleaned_mask = largest_component_mask(closed)

    # Edge detection (for visualization)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Save outputs
    rel_name = in_path.name  # e.g. "image1.jpg"
    base_name = os.path.splitext(rel_name)[0]

    preproc_out = PREPROCESSED_DIR / split / cls / f"{base_name}.png"
    mask_out = MASKS_DIR / split / cls / f"{base_name}_mask.png"
    edges_out = EDGES_DIR / split / cls / f"{base_name}_edges.png"

    # Preprocessed image = resized, normalized to 0-255
    # (CNN will later re-normalize as tensors)
    cv2.imwrite(str(preproc_out), img_resized)
    cv2.imwrite(str(mask_out), cleaned_mask)
    cv2.imwrite(str(edges_out), edges)


def main():
    ensure_dirs()
    for split in SPLITS:
        for cls in CLASSES:
            in_dir = DATA_DIR / split / cls
            if not in_dir.exists():
                print(f"Skipping missing folder: {in_dir}")
                continue

            for fname in os.listdir(in_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    in_path = in_dir / fname
                    process_single_image(in_path, split, cls)
    print("Preprocessing done.")


if __name__ == "__main__":
    main()
