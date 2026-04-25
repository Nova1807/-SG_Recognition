"""Recording mode for signs with insufficient training data."""

import time

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    MIN_VIDEOS_PER_SIGN,
    RECORDING_DURATION_SECONDS,
    RECORDING_FPS,
    RECORDINGS_DIR,
    WEBCAM_INDEX,
    ensure_dirs,
)
from src.scraper import _sanitize_dirname, get_recording_count, get_video_count

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_signs_needing_data(sign_list: list[str]) -> list[dict[str, object]]:
    """Find signs that need more training data."""
    needs_data = []
    for sign_name in sign_list:
        video_count = get_video_count(sign_name)
        recording_count = get_recording_count(sign_name)
        total = video_count + recording_count

        if total < MIN_VIDEOS_PER_SIGN:
            needs_data.append({
                "name": sign_name,
                "videos": video_count,
                "recordings": recording_count,
                "total": total,
                "needed": MIN_VIDEOS_PER_SIGN - total,
            })

    return needs_data


def run_recording_mode(sign_list: list[str]) -> None:
    """Run the interactive recording mode."""
    ensure_dirs()

    signs_needing_data = get_signs_needing_data(sign_list)

    if not signs_needing_data:
        print("\nAlle Gebärden haben genug Trainingsdaten!")
        return

    print("\n" + "=" * 60)
    print("AUFNAHME-MODUS")
    print("=" * 60)
    print(f"\n{len(signs_needing_data)} Gebärden brauchen mehr Trainingsdaten:\n")

    for i, sign_info in enumerate(signs_needing_data):
        print(
            f"  {i + 1}. {sign_info['name']}: "
            f"{sign_info['total']}/{MIN_VIDEOS_PER_SIGN} vorhanden "
            f"(noch {sign_info['needed']} benötigt)"
        )

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("\n[FEHLER] Webcam konnte nicht geöffnet werden!")
        return

    current_sign_idx = 0

    print("\nSteuertasten:")
    print("  LEERTASTE = Aufnahme starten")
    print("  N = Nächste Gebärde")
    print("  P = Vorherige Gebärde")
    print("  Q = Beenden")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened() and current_sign_idx < len(signs_needing_data):
            sign_info = signs_needing_data[current_sign_idx]
            sign_name = sign_info["name"]

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            _draw_landmarks_on_frame(display_frame, results)

            _draw_recording_ui(
                display_frame, sign_name, sign_info, current_sign_idx,
                len(signs_needing_data),
            )

            cv2.imshow("OeGS Aufnahme-Modus", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                _record_sign(cap, holistic, sign_name)
                sign_info["recordings"] = get_recording_count(sign_name)
                sign_info["total"] = sign_info["videos"] + sign_info["recordings"]
                sign_info["needed"] = max(0, MIN_VIDEOS_PER_SIGN - sign_info["total"])
            elif key == ord("n"):
                current_sign_idx = min(
                    current_sign_idx + 1, len(signs_needing_data) - 1
                )
            elif key == ord("p"):
                current_sign_idx = max(current_sign_idx - 1, 0)

    cap.release()
    cv2.destroyAllWindows()
    print("\nAufnahme-Modus beendet.")


def _record_sign(
    cap: cv2.VideoCapture,
    holistic: mp_holistic.Holistic,
    sign_name: str,
) -> None:
    """Record a single sign performance."""
    safe_name = _sanitize_dirname(sign_name)
    sign_dir = RECORDINGS_DIR / safe_name
    sign_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    save_path = sign_dir / f"recording_{timestamp}.mp4"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(save_path), fourcc, RECORDING_FPS, (frame_width, frame_height))

    total_frames = RECORDING_FPS * RECORDING_DURATION_SECONDS
    countdown_frames = RECORDING_FPS * 3

    print(f"\n  Aufnahme für '{sign_name}' startet in 3 Sekunden...")

    for i in range(countdown_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        seconds_left = 3 - (i // RECORDING_FPS)
        cv2.rectangle(display, (0, 0), (frame_width, 100), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"Vorbereitung: {seconds_left}",
            (frame_width // 2 - 150, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            display,
            f"Gebärde: {sign_name}",
            (frame_width // 2 - 100, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.imshow("OeGS Aufnahme-Modus", display)
        cv2.waitKey(int(1000 / RECORDING_FPS))

    print(f"  AUFNAHME LÄUFT... ({RECORDING_DURATION_SECONDS}s)")

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        out.write(frame)

        display = frame.copy()
        progress = (i + 1) / total_frames
        bar_width = int(frame_width * progress)

        cv2.rectangle(display, (0, 0), (frame_width, 60), (0, 0, 0), -1)
        cv2.rectangle(display, (0, frame_height - 10), (bar_width, frame_height), (0, 0, 255), -1)
        cv2.putText(
            display,
            f"AUFNAHME: {sign_name} ({progress:.0%})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.circle(display, (frame_width - 30, 30), 10, (0, 0, 255), -1)

        cv2.imshow("OeGS Aufnahme-Modus", display)
        cv2.waitKey(int(1000 / RECORDING_FPS))

    out.release()
    print(f"  Aufnahme gespeichert: {save_path}")


def _draw_landmarks_on_frame(frame: np.ndarray, results: object) -> None:
    """Draw hand and pose landmarks on frame."""
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


def _draw_recording_ui(
    frame: np.ndarray,
    sign_name: str,
    sign_info: dict[str, object],
    current_idx: int,
    total_signs: int,
) -> None:
    """Draw the recording mode UI overlay."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Gebärde: {sign_name}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    needed = sign_info["needed"]
    total = sign_info["total"]
    status_text = f"Vorhanden: {total}/{MIN_VIDEOS_PER_SIGN}"
    if needed > 0:
        status_text += f" (noch {needed} benötigt)"
    else:
        status_text += " (genug!)"

    status_color = (0, 255, 0) if needed <= 0 else (0, 255, 255)
    cv2.putText(
        frame, status_text, (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1,
    )

    nav_text = f"Gebärde {current_idx + 1}/{total_signs}"
    cv2.putText(
        frame, nav_text, (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "LEERTASTE=Aufnahme  N=Nächste  P=Vorherige  Q=Beenden",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )
