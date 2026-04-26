import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.config import (
    FEATURE_VERSION_PATH,
    FEATURES_PER_FRAME,
    FINGER_CURLS_PER_HAND,
    LABEL_ENCODER_PATH,
    LEFT_CURL_IDX,
    LEFT_HAND_END,
    LEFT_HAND_START,
    LEFT_PRESENT_IDX,
    LEFT_WRIST_IDX,
    MODEL_PATH,
    MODELS_DIR,
    PRIMARY_CURL_IDX,
    PRIMARY_HAND_END,
    PRIMARY_HAND_START,
    PRIMARY_PRESENT_IDX,
    PRIMARY_WRIST_IDX,
    RIGHT_CURL_IDX,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
    RIGHT_PRESENT_IDX,
    RIGHT_WRIST_IDX,
    UNKNOWN_SIGN_LABEL,
    ensure_dirs,
)
from src.landmark_extractor import load_training_data


def _augment_sequence(sequence: np.ndarray, n_augments: int = 5) -> list[np.ndarray]:
    augmented = [sequence.astype(np.float32)]
    rng = np.random.default_rng()
    seq_len = sequence.shape[0]

    for k in range(n_augments):
        shift = int(rng.integers(2, 7))
        direction = shift if rng.random() > 0.5 else -shift

        rolled = np.roll(sequence, direction, axis=0).astype(np.float32)
        aug = rolled.copy()

        noise_std = 0.02 + 0.01 * (k % 3)
        aug[:, PRIMARY_HAND_START:] += rng.normal(
            0, noise_std, aug[:, PRIMARY_HAND_START:].shape
        ).astype(np.float32)
        aug[:, :PRIMARY_HAND_START] = rolled[:, :PRIMARY_HAND_START]
        augmented.append(aug)

    speed_factor = rng.uniform(0.7, 0.9)
    new_len = max(5, int(seq_len * speed_factor))
    indices = np.linspace(0, seq_len - 1, new_len).astype(int)
    fast = sequence[indices]
    last_row = fast[-1:]
    while len(fast) < seq_len:
        fast = np.concatenate([fast, last_row], axis=0)
    augmented.append(fast[:seq_len].astype(np.float32))

    speed_factor = rng.uniform(1.1, 1.4)
    new_len = int(seq_len * speed_factor)
    indices = np.linspace(0, seq_len - 1, new_len).astype(int)
    slow = sequence[indices]
    step = len(slow) / seq_len
    pick = [int(i * step) for i in range(seq_len)]
    augmented.append(slow[pick].astype(np.float32))

    scale = float(rng.uniform(0.8, 1.2))
    scaled = sequence.copy().astype(np.float32)
    scaled[:, PRIMARY_HAND_START:PRIMARY_HAND_END] *= scale
    scaled[:, LEFT_HAND_START:LEFT_HAND_END] *= scale
    scaled[:, RIGHT_HAND_START:RIGHT_HAND_END] *= scale
    augmented.append(scaled)

    return augmented


def _mirror_hands(sequence: np.ndarray) -> np.ndarray:
    """Swap left and right hand features so the model treats both hands equally."""
    aug = sequence.copy().astype(np.float32)

    left_present = sequence[:, LEFT_PRESENT_IDX].copy()
    right_present = sequence[:, RIGHT_PRESENT_IDX].copy()
    aug[:, LEFT_PRESENT_IDX] = right_present
    aug[:, RIGHT_PRESENT_IDX] = left_present

    left_hand = sequence[:, LEFT_HAND_START:LEFT_HAND_END].copy()
    right_hand = sequence[:, RIGHT_HAND_START:RIGHT_HAND_END].copy()
    aug[:, LEFT_HAND_START:LEFT_HAND_END] = right_hand
    aug[:, RIGHT_HAND_START:RIGHT_HAND_END] = left_hand

    left_wrist = sequence[:, LEFT_WRIST_IDX:LEFT_WRIST_IDX + 2].copy()
    right_wrist = sequence[:, RIGHT_WRIST_IDX:RIGHT_WRIST_IDX + 2].copy()
    aug[:, LEFT_WRIST_IDX] = 1.0 - right_wrist[:, 0]
    aug[:, LEFT_WRIST_IDX + 1] = right_wrist[:, 1]
    aug[:, RIGHT_WRIST_IDX] = 1.0 - left_wrist[:, 0]
    aug[:, RIGHT_WRIST_IDX + 1] = left_wrist[:, 1]

    aug[:, PRIMARY_WRIST_IDX] = 1.0 - sequence[:, PRIMARY_WRIST_IDX]

    left_curls = sequence[:, LEFT_CURL_IDX:LEFT_CURL_IDX + FINGER_CURLS_PER_HAND].copy()
    right_curls = sequence[:, RIGHT_CURL_IDX:RIGHT_CURL_IDX + FINGER_CURLS_PER_HAND].copy()
    aug[:, LEFT_CURL_IDX:LEFT_CURL_IDX + FINGER_CURLS_PER_HAND] = right_curls
    aug[:, RIGHT_CURL_IDX:RIGHT_CURL_IDX + FINGER_CURLS_PER_HAND] = left_curls

    return aug


def _build_features(X_all: np.ndarray) -> np.ndarray:
    n_samples, seq_len, features = X_all.shape
    X_flat = X_all.reshape(n_samples, seq_len * features)
    X_mean = np.mean(X_all, axis=1)
    X_std = np.std(X_all, axis=1)
    X_max = np.max(X_all, axis=1)
    X_min = np.min(X_all, axis=1)

    X_diff = np.diff(X_all, axis=1, prepend=X_all[:, :1, :])
    X_vel_mean = np.mean(np.abs(X_diff), axis=1)
    X_vel_max = np.max(np.abs(X_diff), axis=1)

    return np.hstack([X_flat, X_mean, X_std, X_max, X_min, X_vel_mean, X_vel_max])


def _is_left_only_sequence(sequence: np.ndarray) -> bool:
    has_left = bool(np.any(sequence[:, LEFT_PRESENT_IDX] > 0.5))
    has_right = bool(np.any(sequence[:, RIGHT_PRESENT_IDX] > 0.5))
    return has_left and not has_right


def _is_right_only_sequence(sequence: np.ndarray) -> bool:
    has_left = bool(np.any(sequence[:, LEFT_PRESENT_IDX] > 0.5))
    has_right = bool(np.any(sequence[:, RIGHT_PRESENT_IDX] > 0.5))
    return has_right and not has_left


def _inject_other_hand(
    sequence: np.ndarray,
    donor_sequence: np.ndarray,
    target: str,
) -> np.ndarray:
    """Inject a donor hand into the empty hand slot (including curls/wrists)."""
    aug = sequence.copy().astype(np.float32)
    donor_primary = donor_sequence[:, PRIMARY_HAND_START:PRIMARY_HAND_END]
    donor_curls = donor_sequence[:, PRIMARY_CURL_IDX:PRIMARY_CURL_IDX + FINGER_CURLS_PER_HAND]
    donor_wrist = donor_sequence[:, PRIMARY_WRIST_IDX:PRIMARY_WRIST_IDX + 2]

    donor_has_primary = (
        np.linalg.norm(donor_primary, axis=1) > 1e-6
    ).astype(np.float32)

    if target == "left":
        aug[:, LEFT_PRESENT_IDX] = np.maximum(
            aug[:, LEFT_PRESENT_IDX], donor_has_primary
        )
        aug[:, LEFT_HAND_START:LEFT_HAND_END] = donor_primary
        aug[:, LEFT_CURL_IDX:LEFT_CURL_IDX + FINGER_CURLS_PER_HAND] = donor_curls
        aug[:, LEFT_WRIST_IDX:LEFT_WRIST_IDX + 2] = donor_wrist
    else:
        aug[:, RIGHT_PRESENT_IDX] = np.maximum(
            aug[:, RIGHT_PRESENT_IDX], donor_has_primary
        )
        aug[:, RIGHT_HAND_START:RIGHT_HAND_END] = donor_primary
        aug[:, RIGHT_CURL_IDX:RIGHT_CURL_IDX + FINGER_CURLS_PER_HAND] = donor_curls
        aug[:, RIGHT_WRIST_IDX:RIGHT_WRIST_IDX + 2] = donor_wrist

    return aug


def train_model(sign_list: list[str]) -> bool:
    ensure_dirs()
    print("\nLade Trainingsdaten...")
    X, y, label_names = load_training_data(sign_list)

    if len(X) == 0 or len(label_names) < 2:
        print("[FEHLER] Zu wenig Trainingsdaten!")
        return False

    print(f"  {len(X)} Sequenzen, {len(label_names)} Gebaerden")

    real_hand_sequences = X.copy().astype(np.float32)
    unknown_label_idx = len(label_names)

    rng_noise = np.random.default_rng(123)
    n_noise = max(1, len(X) // 3)
    noise_unknown = rng_noise.normal(0, 0.05, (n_noise,) + X.shape[1:]).astype(
        np.float32
    )
    noise_unknown[:, :, :PRIMARY_HAND_START] = 0.0

    unknown_zero = np.zeros_like(X[: max(1, len(X) // 4)], dtype=np.float32)
    unknown = np.concatenate([unknown_zero, noise_unknown], axis=0)

    X = np.concatenate([X, unknown], axis=0).astype(np.float32)
    y = np.concatenate(
        [y, np.full(len(unknown), unknown_label_idx, dtype=np.int64)],
        axis=0,
    )
    label_names = list(label_names) + [UNKNOWN_SIGN_LABEL]

    rng = np.random.default_rng(42)
    donor_pool = [
        seq
        for seq in real_hand_sequences
        if np.any(seq[:, PRIMARY_HAND_START:PRIMARY_HAND_END] != 0)
    ]

    X_aug, y_aug = [], []
    for i in range(len(X)):
        augmented_versions = _augment_sequence(X[i], n_augments=6)

        for seq in augmented_versions:
            X_aug.append(seq)
            y_aug.append(y[i])

            if y[i] == unknown_label_idx:
                continue

            X_aug.append(_mirror_hands(seq))
            y_aug.append(y[i])

            if not donor_pool:
                continue

            if _is_left_only_sequence(seq) or _is_right_only_sequence(seq):
                target = "right" if _is_left_only_sequence(seq) else "left"
                for _ in range(3):
                    donor = donor_pool[int(rng.integers(0, len(donor_pool)))]
                    X_aug.append(_inject_other_hand(seq, donor, target=target))
                    y_aug.append(y[i])

    X_all = np.array(X_aug, dtype=np.float32)
    y_all = np.array(y_aug, dtype=np.int64)

    print(f"  Augmentiert: {len(X_all)} Samples")

    le = LabelEncoder()
    le.fit(label_names)
    y_encoded = le.transform([label_names[i] for i in y_all])

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    X_feat = _build_features(X_all)

    n_folds = min(5, min(np.bincount(y_encoded)))
    if n_folds >= 2:
        scores = cross_val_score(clf, X_feat, y_encoded, cv=n_folds, scoring="accuracy")
        print(f"  Cross-Validation Accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")
    else:
        print("  (Zu wenig Daten fuer Cross-Validation)")

    clf.fit(X_feat, y_encoded)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    FEATURE_VERSION_PATH.write_text(str(FEATURES_PER_FRAME))
    print(f"  Modell gespeichert (Feature-Version: {FEATURES_PER_FRAME})")
    return True


def predict_with_features(
    clf: RandomForestClassifier, sequence: np.ndarray
) -> np.ndarray:
    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, :]
    return clf.predict_proba(_build_features(sequence.astype(np.float32)))


def _zero_hand_slot(
    seq: np.ndarray, side: str,
) -> np.ndarray:
    """Zero out one hand's features (landmarks, wrist, curls)."""
    out = seq.copy()
    if side == "left":
        out[:, :, LEFT_PRESENT_IDX] = 0
        out[:, :, LEFT_HAND_START:LEFT_HAND_END] = 0
        out[:, :, LEFT_WRIST_IDX:LEFT_WRIST_IDX + 2] = 0
        out[:, :, LEFT_CURL_IDX:LEFT_CURL_IDX + FINGER_CURLS_PER_HAND] = 0
    else:
        out[:, :, RIGHT_PRESENT_IDX] = 0
        out[:, :, RIGHT_HAND_START:RIGHT_HAND_END] = 0
        out[:, :, RIGHT_WRIST_IDX:RIGHT_WRIST_IDX + 2] = 0
        out[:, :, RIGHT_CURL_IDX:RIGHT_CURL_IDX + FINGER_CURLS_PER_HAND] = 0
    return out


def predict_best_hand(
    clf: RandomForestClassifier, sequence: np.ndarray,
) -> np.ndarray:
    """When both hands visible, try full / right-only / left-only; return best."""
    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, :]
    seq = sequence.astype(np.float32)

    proba_full = clf.predict_proba(_build_features(seq))[0]

    has_left = bool(np.any(seq[:, :, LEFT_PRESENT_IDX] > 0.5))
    has_right = bool(np.any(seq[:, :, RIGHT_PRESENT_IDX] > 0.5))

    if not (has_left and has_right):
        return proba_full[np.newaxis, :]

    candidates = [proba_full]

    right_focus = _zero_hand_slot(seq, "left")
    right_focus[:, :, PRIMARY_PRESENT_IDX] = right_focus[:, :, RIGHT_PRESENT_IDX]
    right_focus[:, :, PRIMARY_HAND_START:PRIMARY_HAND_END] = (
        right_focus[:, :, RIGHT_HAND_START:RIGHT_HAND_END]
    )
    right_focus[:, :, PRIMARY_WRIST_IDX:PRIMARY_WRIST_IDX + 2] = (
        right_focus[:, :, RIGHT_WRIST_IDX:RIGHT_WRIST_IDX + 2]
    )
    right_focus[:, :, PRIMARY_CURL_IDX:PRIMARY_CURL_IDX + FINGER_CURLS_PER_HAND] = (
        right_focus[:, :, RIGHT_CURL_IDX:RIGHT_CURL_IDX + FINGER_CURLS_PER_HAND]
    )
    candidates.append(clf.predict_proba(_build_features(right_focus))[0])

    left_focus = _zero_hand_slot(seq, "right")
    left_focus[:, :, PRIMARY_PRESENT_IDX] = left_focus[:, :, LEFT_PRESENT_IDX]
    left_focus[:, :, PRIMARY_HAND_START:PRIMARY_HAND_END] = (
        left_focus[:, :, LEFT_HAND_START:LEFT_HAND_END]
    )
    left_focus[:, :, PRIMARY_WRIST_IDX:PRIMARY_WRIST_IDX + 2] = (
        left_focus[:, :, LEFT_WRIST_IDX:LEFT_WRIST_IDX + 2]
    )
    left_focus[:, :, PRIMARY_CURL_IDX:PRIMARY_CURL_IDX + FINGER_CURLS_PER_HAND] = (
        left_focus[:, :, LEFT_CURL_IDX:LEFT_CURL_IDX + FINGER_CURLS_PER_HAND]
    )
    candidates.append(clf.predict_proba(_build_features(left_focus))[0])

    best = max(candidates, key=lambda p: float(p.max()))
    return best[np.newaxis, :]


def load_model() -> tuple[RandomForestClassifier, LabelEncoder] | None:
    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        print("[FEHLER] Kein trainiertes Modell gefunden!")
        return None
    return joblib.load(MODEL_PATH), joblib.load(LABEL_ENCODER_PATH)
