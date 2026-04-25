"""Extract hand and pose landmarks from sign language videos using MediaPipe."""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import (
    LANDMARKS_DIR,
    LANDMARK_SEQUENCE_LENGTH,
    RECORDINGS_DIR,
    VIDEOS_DIR,
    ensure_dirs,
)
from src.mediapipe_compat import HolisticDetector, extract_landmarks_from_result
from src.scraper import _sanitize_dirname


def extract_landmarks_from_frame(
    frame: np.ndarray, detector: HolisticDetector
) -> np.ndarray | None:
    """Extract hand and pose landmarks from a single frame.

    Returns a flattened numpy array of landmarks, or None if no landmarks detected.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(image_rgb)
    return extract_landmarks_from_result(result)


def extract_landmarks_from_video(video_path: Path) -> list[np.ndarray]:
    """Extract landmark sequences from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [FEHLER] Video konnte nicht geöffnet werden: {video_path}")
        return []

    frames_landmarks = []

    with HolisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = extract_landmarks_from_frame(frame, detector)
            if landmarks is not None:
                frames_landmarks.append(landmarks)

    cap.release()
    return frames_landmarks


def pad_or_truncate_sequence(
    sequence: list[np.ndarray], target_length: int
) -> np.ndarray:
    """Pad or truncate a landmark sequence to a fixed length."""
    if len(sequence) == 0:
        feature_size = 258  # 63 + 63 + 132
        return np.zeros((target_length, feature_size))

    feature_size = sequence[0].shape[0]

    if len(sequence) >= target_length:
        step = len(sequence) / target_length
        indices = [int(i * step) for i in range(target_length)]
        return np.array([sequence[idx] for idx in indices])
    else:
        padded = list(sequence)
        while len(padded) < target_length:
            padded.append(np.zeros(feature_size))
        return np.array(padded)


def extract_all_landmarks(sign_list: list[str]) -> None:
    """Extract landmarks from all videos and recordings for all signs."""
    ensure_dirs()

    for sign_name in sign_list:
        print(f"\nExtrahiere Landmarks für: {sign_name}")
        safe_name = _sanitize_dirname(sign_name)

        video_dir = VIDEOS_DIR / safe_name
        recording_dir = RECORDINGS_DIR / safe_name
        landmark_dir = LANDMARKS_DIR / safe_name
        landmark_dir.mkdir(parents=True, exist_ok=True)

        video_files: list[Path] = []
        if video_dir.exists():
            video_files.extend(video_dir.glob("*.mp4"))
        if recording_dir.exists():
            video_files.extend(recording_dir.glob("*.mp4"))

        if not video_files:
            print(f"  Keine Videos für '{sign_name}' gefunden.")
            continue

        for video_path in tqdm(
            video_files, desc=f"  '{sign_name}'", unit="video"
        ):
            landmark_file = landmark_dir / f"{video_path.stem}.npy"
            if landmark_file.exists():
                continue

            landmarks = extract_landmarks_from_video(video_path)
            if not landmarks:
                print(f"  [WARNUNG] Keine Landmarks in: {video_path.name}")
                continue

            sequence = pad_or_truncate_sequence(
                landmarks, LANDMARK_SEQUENCE_LENGTH
            )
            np.save(landmark_file, sequence)

        total = len(list(landmark_dir.glob("*.npy")))
        print(f"  Gesamt: {total} Landmark-Dateien für '{sign_name}'")


def load_training_data(
    sign_list: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all landmark data for training.

    Returns (X, y, labels) where:
    - X: array of shape (n_samples, sequence_length, features)
    - y: array of integer labels
    - labels: list of sign names corresponding to label indices
    """
    all_sequences = []
    all_labels = []
    label_names = []

    for sign_name in sign_list:
        safe_name = _sanitize_dirname(sign_name)
        landmark_dir = LANDMARKS_DIR / safe_name

        if not landmark_dir.exists():
            continue

        npy_files = list(landmark_dir.glob("*.npy"))
        if not npy_files:
            continue

        label_idx = len(label_names)
        label_names.append(sign_name)

        for npy_file in npy_files:
            sequence = np.load(npy_file)
            all_sequences.append(sequence)
            all_labels.append(label_idx)

    if not all_sequences:
        return np.array([]), np.array([]), []

    return np.array(all_sequences), np.array(all_labels), label_names
