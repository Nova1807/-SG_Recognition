"""Live webcam sign language recognition."""

from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    LANDMARK_SEQUENCE_LENGTH,
    WEBCAM_INDEX,
)
from src.landmark_extractor import extract_landmarks_from_frame
from src.trainer import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


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

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            landmarks = extract_landmarks_from_frame(frame, holistic)
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
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)

            _draw_landmarks(display_frame, results)

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


def _draw_landmarks(frame: np.ndarray, results: object) -> None:
    """Draw hand and pose landmarks on the frame."""
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )


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

    if prediction:
        text = f"Erkannt: {prediction} ({confidence:.0%})"
        color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255)
        cv2.putText(
            frame, text, (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

    status_color = (0, 0, 255) if is_recording else (128, 128, 128)
    status_text = f"Aufnahme... ({buffer_size} Frames)" if is_recording else "Warte auf Gebärde..."
    cv2.putText(
        frame, status_text, (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1,
    )

    cv2.putText(
        frame, "Q=Beenden  R=Reset", (w - 200, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
    )
