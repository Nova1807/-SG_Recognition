"""Live webcam sign language recognition."""

from collections import deque

import cv2
import numpy as np

from src.config import (
    LANDMARK_SEQUENCE_LENGTH,
    WEBCAM_INDEX,
)
from src.landmark_extractor import extract_landmarks_from_frame
from src.mediapipe_compat import HolisticDetector, draw_landmarks_on_frame
from src.trainer import load_model


def run_recognition() -> None:
    """Start live webcam recognition."""
    result = load_model()
    if result is None:
        return

    clf, le = result

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[FEHLER] Webcam konnte nicht geöffnet werden!")
        return

    print("\n" + "=" * 50)
    print("LIVE GEBÄRDEN-ERKENNUNG")
    print("=" * 50)
    print("Drücke 'q' zum Beenden")
    print("Drücke 'r' um die Erkennung zurückzusetzen")
    print()

    landmark_buffer: deque[np.ndarray] = deque(maxlen=LANDMARK_SEQUENCE_LENGTH)
    current_prediction = ""
    confidence = 0.0
    is_recording_gesture = False

    with HolisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            landmarks = extract_landmarks_from_frame(frame, detector)
            if landmarks is not None:
                hand_data = landmarks[:126]
                has_hands = np.any(hand_data != 0)

                if has_hands:
                    if not is_recording_gesture:
                        is_recording_gesture = True
                        landmark_buffer.clear()
                    landmark_buffer.append(landmarks)
                else:
                    if is_recording_gesture and len(landmark_buffer) >= 5:
                        prediction, conf = _predict_sign(
                            list(landmark_buffer), clf, le
                        )
                        if prediction:
                            current_prediction = prediction
                            confidence = conf
                    is_recording_gesture = False

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_lm = detector.process(image_rgb)
            draw_landmarks_on_frame(display_frame, result_lm)

            _draw_ui(display_frame, current_prediction, confidence,
                     len(landmark_buffer), is_recording_gesture)

            cv2.imshow("OeGS Erkennung", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                landmark_buffer.clear()
                current_prediction = ""
                confidence = 0.0

    cap.release()
    cv2.destroyAllWindows()


def _predict_sign(
    sequence: list[np.ndarray],
    clf: object,
    le: object,
) -> tuple[str, float]:
    """Predict a sign from a landmark sequence."""
    from src.landmark_extractor import pad_or_truncate_sequence

    padded = pad_or_truncate_sequence(sequence, LANDMARK_SEQUENCE_LENGTH)
    flat = padded.flatten().reshape(1, -1)

    proba = clf.predict_proba(flat)[0]
    max_idx = np.argmax(proba)
    conf = proba[max_idx]

    if conf < 0.3:
        return "", 0.0

    prediction = le.inverse_transform([max_idx])[0]
    return prediction, conf


def _draw_ui(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    buffer_size: int,
    is_recording: bool,
) -> None:
    """Draw the UI overlay on the frame."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "OeGS Gebärden-Erkennung",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    status = "Aufnahme..." if is_recording else "Warte auf Gebärde..."
    color = (0, 0, 255) if is_recording else (0, 255, 0)
    cv2.putText(
        frame,
        f"Status: {status} ({buffer_size} Frames)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )

    if prediction:
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
        conf_text = f"Konfidenz: {confidence:.0%}"
        cv2.putText(
            frame,
            conf_text,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
