from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    DISPLAY_CONFIDENCE_THRESHOLD,
    LANDMARK_SEQUENCE_LENGTH,
    LEFT_PRESENT_IDX,
    PRIMARY_PRESENT_IDX,
    REJECT_CONFIDENCE_THRESHOLD,
    RIGHT_PRESENT_IDX,
    WEBCAM_INDEX,
)
from src.landmark_extractor import (
    extract_landmarks_from_video,
    pad_or_truncate_sequence,
)
from src.mediapipe_compat import (
    HolisticDetector,
    draw_landmarks_on_frame,
    extract_landmarks_from_result,
)
from src.trainer import load_model, predict_best_hand, predict_with_features


def predict_sequence(sequence: list[np.ndarray] | np.ndarray) -> dict[str, object]:
    result = load_model()
    if result is None:
        return {"error": "Kein Modell geladen"}

    clf, le = result

    if isinstance(sequence, list):
        if not sequence:
            return {"prediction": None, "confidence": 0.0}
        sequence = pad_or_truncate_sequence(sequence, LANDMARK_SEQUENCE_LENGTH)

    proba = predict_best_hand(clf, sequence)[0]
    max_idx = int(np.argmax(proba))
    conf = float(proba[max_idx])
    pred = le.inverse_transform([max_idx])[0]

    if conf < REJECT_CONFIDENCE_THRESHOLD or pred == "unknown":
        return {"prediction": None, "confidence": 0.0}

    return {"prediction": pred, "confidence": conf}


def predict_video_file(video_path: str | Path) -> dict[str, object]:
    video_path = Path(video_path)
    landmarks = extract_landmarks_from_video(video_path)

    if not landmarks:
        return {
            "prediction": None,
            "confidence": 0.0,
            "frames_used": 0,
            "error": "Keine Hand erkannt",
        }

    result = predict_sequence(landmarks)
    result["frames_used"] = len(landmarks)
    return result


def run_recognition() -> None:
    result = load_model()
    if result is None:
        return
    clf, le = result

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[FEHLER] Webcam konnte nicht geöffnet werden!")
        return

    landmark_buffer: deque[np.ndarray] = deque(maxlen=LANDMARK_SEQUENCE_LENGTH)
    current_prediction = ""
    confidence = 0.0

    with HolisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_lm = detector.process(image_rgb)
            landmarks = extract_landmarks_from_result(result_lm)

            has_hands = bool(
                landmarks[PRIMARY_PRESENT_IDX]
                or landmarks[LEFT_PRESENT_IDX]
                or landmarks[RIGHT_PRESENT_IDX]
            )

            if has_hands:
                landmark_buffer.append(landmarks.astype(np.float32))

                if len(landmark_buffer) >= 10:
                    padded = pad_or_truncate_sequence(
                        list(landmark_buffer),
                        LANDMARK_SEQUENCE_LENGTH,
                    )
                    proba = predict_best_hand(clf, padded)[0]
                    max_idx = int(np.argmax(proba))
                    conf = float(proba[max_idx])
                    pred = le.inverse_transform([max_idx])[0]

                    if conf >= REJECT_CONFIDENCE_THRESHOLD and pred != "unknown":
                        current_prediction = pred
                        confidence = conf
                    else:
                        current_prediction = ""
                        confidence = 0.0
            else:
                landmark_buffer.clear()
                current_prediction = ""
                confidence = 0.0

            display_frame = frame.copy()
            draw_landmarks_on_frame(display_frame, result_lm)
            display_frame = cv2.flip(display_frame, 1)

            _draw_ui(
                display_frame,
                current_prediction,
                confidence,
                len(landmark_buffer),
            )

            cv2.imshow("OeGS Erkennung", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                landmark_buffer.clear()
                current_prediction = ""
                confidence = 0.0

    cap.release()
    cv2.destroyAllWindows()


def _draw_ui(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    buffer_size: int,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "OeGS Gebaerden-Erkennung",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Frames: {buffer_size}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    if prediction and confidence >= DISPLAY_CONFIDENCE_THRESHOLD:
        cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            prediction,
            (10, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            frame,
            f"Konfidenz: {confidence:.0%}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
