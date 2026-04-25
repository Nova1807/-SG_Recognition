"""Train a sign language recognition model."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import joblib

from src.config import (
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    MODELS_DIR,
    ensure_dirs,
)
from src.landmark_extractor import load_training_data


def train_model(sign_list: list[str]) -> bool:
    """Train a RandomForest classifier on the extracted landmarks.

    Returns True if training was successful.
    """
    ensure_dirs()

    print("\nLade Trainingsdaten...")
    X, y, label_names = load_training_data(sign_list)

    if len(X) == 0:
        print("[FEHLER] Keine Trainingsdaten vorhanden!")
        print("Bitte zuerst Videos herunterladen und Landmarks extrahieren.")
        return False

    print(f"  Samples: {len(X)}")
    print(f"  Gebärden: {len(label_names)}")
    for i, name in enumerate(label_names):
        count = np.sum(y == i)
        print(f"    {name}: {count} Samples")

    if len(label_names) < 2:
        print("[FEHLER] Mindestens 2 verschiedene Gebärden benötigt!")
        return False

    n_samples, seq_len, features = X.shape
    X_flat = X.reshape(n_samples, seq_len * features)

    le = LabelEncoder()
    le.fit(label_names)
    y_encoded = le.transform([label_names[i] for i in y])

    print("\nTrainiere Modell...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    if len(X_flat) >= 5:
        n_splits = min(5, min(np.bincount(y_encoded)))
        if n_splits >= 2:
            scores = cross_val_score(clf, X_flat, y_encoded, cv=n_splits)
            print(f"  Cross-Validation Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

    clf.fit(X_flat, y_encoded)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print(f"\nModell gespeichert: {MODEL_PATH}")
    print(f"Label-Encoder gespeichert: {LABEL_ENCODER_PATH}")
    return True


def load_model() -> tuple[RandomForestClassifier, LabelEncoder] | None:
    """Load the trained model and label encoder."""
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        print("[FEHLER] Kein trainiertes Modell gefunden!")
        print("Bitte zuerst 'python -m src.main train' ausführen.")
        return None

    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return clf, le
