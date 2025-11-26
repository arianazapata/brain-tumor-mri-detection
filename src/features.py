# src/features.py
import os
import numpy as np
import cv2
from pathlib import Path

SPLITS = ["train", "val", "test"]
CLASSES = ["tumor", "no_tumor"]

PREPROCESSED_DIR = Path("preprocessed")
MASKS_DIR = Path("results") / "masks"
FEATURES_DIR = Path("results") / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"no_tumor": 0, "tumor": 1}


def extract_features_for_image(gray_img: np.ndarray, mask: np.ndarray):
    # Ensure binary mask
    mask_bin = (mask > 0).astype(np.uint8)

    area = int(np.count_nonzero(mask_bin))
    if area > 0:
        ys, xs = np.where(mask_bin > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        width = int(x_max - x_min + 1)
        height = int(y_max - y_min + 1)
        aspect_ratio = float(width) / float(height) if height > 0 else 0.0
        mean_intensity = float(gray_img[mask_bin > 0].mean())
    else:
        width = height = 0
        aspect_ratio = 0.0
        mean_intensity = float(gray_img.mean())

    # Simple feature vector
    return np.array([area, width, height, aspect_ratio, mean_intensity], dtype=np.float32)


def process_split(split: str):
    X_list = []
    y_list = []

    for cls in CLASSES:
        preproc_dir = PREPROCESSED_DIR / split / cls
        mask_dir = MASKS_DIR / split / cls
        if not preproc_dir.exists():
            print(f"Skipping missing folder: {preproc_dir}")
            continue

        for fname in os.listdir(preproc_dir):
            if not fname.lower().endswith(".png"):
                continue
            base = os.path.splitext(fname)[0]
            gray_path = preproc_dir / fname
            mask_path = mask_dir / f"{base}_mask.png"

            gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"Could not read image: {gray_path}")
                continue

            if not mask_path.exists():
                print(f"Missing mask for {gray_path}, skipping.")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            feats = extract_features_for_image(gray, mask)
            X_list.append(feats)
            y_list.append(LABEL_MAP[cls])

    if not X_list:
        print(f"No features for split {split}")
        return

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    out_path = FEATURES_DIR / f"classical_features_{split}.npz"
    np.savez(out_path, X=X, y=y)
    print(f"Saved {split} features to {out_path} (X.shape={X.shape}, y.shape={y.shape})")


def main():
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
