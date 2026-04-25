"""Train a sign language recognition model."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import joblib

from src.config import (
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    MODELS_DIR,
    ensure_dirs,
)
from src.landmark_extractor import load_training_data


def _augment_sequence(sequence: np.ndarray, n_augments: int = 5) -> list[np.ndarray]:
    """Create augmented versions of a landmark sequence.

    Each frame has 258 features:
    - [0:63] left hand (21 landmarks * xyz)
    - [63:126] right hand (21 landmarks * xyz)
    - [126:258] pose (33 landmarks * xyzv)
    """
    augmented = [sequence]
    rng = np.random.default_rng(42)

    for _ in range(n_augments):
        aug = sequence.copy()

        # Add small noise to coordinates
        noise = rng.normal(0, 0.01, aug.shape).astype(np.float32)
        aug = aug + noise

        # Random time shift: shift sequence by 1-3 frames
        shift = rng.integers(1, 4)
        if rng.random() > 0.5:
            aug = np.roll(aug, shift, axis=0)
        else:
            aug = np.roll(aug, -shift, axis=0)

        # Random speed: subsample or stretch slightly
        if rng.random() > 0.5:
            n_frames = aug.shape[0]
            speed = rng.uniform(0.85, 1.15)
            new_len = max(5, int(n_frames * speed))
            indices = np.linspace(0, n_frames - 1, new_len).astype(int)
            stretched = aug[indices]
            # Pad/truncate back to original length
            if len(stretched) >= n_frames:
                aug = stretched[:n_frames]
            else:
                pad = np.zeros((n_frames - len(stretched), aug.shape[1]))
                aug = np.vstack([stretched, pad])

        augmented.append(aug)

    return augmented


def train_model(sign_list: list[str]) -> bool:
    """Train a RandomForest classifier on the extracted landmarks.

    Returns True if training was successful.
    """
    ensure_dirs()

    print("\nLade Trainingsdaten...")
    X, y, label_names = load_training_data(sign_list)

    if len(X) == 0:
        print("[FEHLER] Keine Trainingsdaten vorhanden!")
        return False

    print(f"  Original Samples: {len(X)}")
    print(f"  Gebaerden: {len(label_names)}")
    for i, name in enumerate(label_names):
        count = int(np.sum(y == i))
        print(f"    {name}: {count} Samples")

    if len(label_names) < 2:
        print("[FEHLER] Mindestens 2 verschiedene Gebaerden benoetigt!")
        return False

    # Augment training data
    print("\nAugmentiere Trainingsdaten...")
    X_aug = []
    y_aug = []
    for i in range(len(X)):
        augmented = _augment_sequence(X[i], n_augments=8)
        for seq in augmented:
            X_aug.append(seq)
            y_aug.append(y[i])

    X_all = np.array(X_aug)
    y_all = np.array(y_aug)
    print(f"  Augmentierte Samples: {len(X_all)}")

    # Use both flattened sequence AND per-frame statistics as features
    n_samples, seq_len, features = X_all.shape
    X_flat = X_all.reshape(n_samples, seq_len * features)

    # Add statistical features: mean, std, min, max per landmark across time
    X_mean = np.mean(X_all, axis=1)
    X_std = np.std(X_all, axis=1)
    X_max = np.max(X_all, axis=1)
    X_min = np.min(X_all, axis=1)
    X_features = np.hstack([X_flat, X_mean, X_std, X_max, X_min])

    le = LabelEncoder()
    le.fit(label_names)
    y_encoded = le.transform([label_names[i] for i in y_all])

    print("\nTrainiere Modell...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_features, y_encoded)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print(f"\nModell gespeichert: {MODEL_PATH}")
    print(f"Label-Encoder gespeichert: {LABEL_ENCODER_PATH}")
    return True


def predict_with_features(
    clf: RandomForestClassifier,
    sequence: np.ndarray,
) -> np.ndarray:
    """Predict using the same feature extraction as training."""
    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, :]

    n_samples, seq_len, features = sequence.shape
    X_flat = sequence.reshape(n_samples, seq_len * features)
    X_mean = np.mean(sequence, axis=1)
    X_std = np.std(sequence, axis=1)
    X_max = np.max(sequence, axis=1)
    X_min = np.min(sequence, axis=1)
    X_features = np.hstack([X_flat, X_mean, X_std, X_max, X_min])

    return clf.predict_proba(X_features)


def load_model() -> tuple[RandomForestClassifier, LabelEncoder] | None:
    """Load the trained model and label encoder."""
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        print("[FEHLER] Kein trainiertes Modell gefunden!")
        return None

    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return clf, le
