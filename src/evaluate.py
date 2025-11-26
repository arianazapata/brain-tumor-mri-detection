# src/evaluate.py
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cnn_model import build_model, DEVICE  # reuse model builder from cnn_model.py

FEATURES_DIR = Path("results") / "features"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSED_DIR = Path("preprocessed")
MODELS_DIR = Path("models")


def load_classical_test():
    path = FEATURES_DIR / "classical_features_test.npz"
    data = np.load(path)
    return data["X"], data["y"]


def eval_classical():
    print("\n=== Evaluating Classical Model on Test Set ===")
    bundle = joblib.load(MODELS_DIR / "classical_model.pkl")
    scaler = bundle["scaler"]
    model = bundle["model"]

    X_test, y_test = load_classical_test()
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=["no_tumor", "tumor"]))
    print("Confusion matrix (classical, test):")
    print(confusion_matrix(y_test, y_pred))


def get_test_loader(batch_size=16):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_dir = PREPROCESSED_DIR / "test"
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader


def eval_cnn():
    print("\n=== Evaluating CNN on Test Set ===")
    dataset, loader = get_test_loader(batch_size=16)
    class_names = dataset.classes  # e.g., ['no_tumor', 'tumor']

    model = build_model(num_classes=2, freeze_base=False)
    state_dict = torch.load(MODELS_DIR / "cnn_best.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion matrix (CNN, test):")
    print(confusion_matrix(all_labels, all_preds))

    # Save metrics as JSON (optional)
    metrics = {
        "cnn_test_report": classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        ),
        "cnn_confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
    with open(RESULTS_DIR / "cnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved CNN metrics to {RESULTS_DIR / 'cnn_metrics.json'}")


def main():
    eval_classical()
    eval_cnn()


if __name__ == "__main__":
    main()
