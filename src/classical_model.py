# src/classical_model.py
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

FEATURES_DIR = Path("results") / "features"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = Path("results") / "classical_metrics.json"


def load_split(split: str):
    path = FEATURES_DIR / f"classical_features_{split}.npz"
    data = np.load(path)
    return data["X"], data["y"]


def main():
    # Load data
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression baseline
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Eval on val
    y_val_pred = clf.predict(X_val_scaled)
    print("=== Classical Model: Validation ===")
    print(classification_report(y_val, y_val_pred, target_names=["no_tumor", "tumor"]))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    # Eval on test
    y_test_pred = clf.predict(X_test_scaled)
    print("=== Classical Model: Test ===")
    print(classification_report(y_test, y_test_pred, target_names=["no_tumor", "tumor"]))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))

    # Save model + scaler
    joblib.dump({"scaler": scaler, "model": clf}, MODELS_DIR / "classical_model.pkl")

    metrics = {
        "val_report": classification_report(
            y_val, y_val_pred, target_names=["no_tumor", "tumor"], output_dict=True
        ),
        "test_report": classification_report(
            y_test, y_test_pred, target_names=["no_tumor", "tumor"], output_dict=True
        ),
        "val_confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist(),
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved classical model + metrics to {MODELS_DIR} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
