"""Desktop popup application for ÖGS sign recognition."""

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    LABEL_ENCODER_PATH,
    LANDMARK_SEQUENCE_LENGTH,
    MIN_VIDEOS_PER_SIGN,
    MODEL_PATH,
    RECORDINGS_DIR,
    RECORDING_DURATION_SECONDS,
    RECORDING_FPS,
    VIDEOS_DIR,
    WEBCAM_INDEX,
    ensure_dirs,
    load_sign_list,
)
from src.mediapipe_compat import HolisticDetector, extract_landmarks_from_result
from src.scraper import (
    _sanitize_dirname,
    get_recording_count,
    get_video_count,
)


_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]


def _draw_landmarks(frame: np.ndarray, result) -> None:
    """Draw hand and pose landmarks on a BGR frame using raw OpenCV."""
    h, w = frame.shape[:2]

    def _draw_hand(landmarks, color, conn_color):
        if not landmarks:
            return
        pts = []
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            pts.append((px, py))
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.circle(frame, (px, py), 5, conn_color, 1)
        for i, j in _HAND_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, pts[i], pts[j], conn_color, 2)

    _draw_hand(result.left_hand_landmarks, (0, 200, 255), (0, 150, 200))
    _draw_hand(result.right_hand_landmarks, (255, 200, 0), (200, 150, 0))

    if result.pose_landmarks:
        pts = []
        pose_landmarks = result.pose_landmarks
        if hasattr(pose_landmarks, "landmark"):
            pose_landmarks = pose_landmarks.landmark
        for lm in pose_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            pts.append((px, py))
            cv2.circle(frame, (px, py), 3, (100, 255, 100), -1)
        for i, j in _POSE_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, pts[i], pts[j], (80, 200, 80), 2)


def _resize_keep_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize frame keeping aspect ratio, pad with black."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


class DesktopApp:
    """Main desktop application for sign recognition."""

    def __init__(self):
        self.sign_list = load_sign_list()
        self.detector = None
        self.clf = None
        self.le = None
        self.pipeline_done = False
        self.pipeline_status = "Starte..."
        self.landmark_buffer: deque[np.ndarray] = deque(maxlen=LANDMARK_SEQUENCE_LENGTH)
        self.current_prediction = ""
        self.confidence = 0.0
        self.last_result = None
        self.frame_count = 0

        self.recording_mode = False
        self.signs_needing_data: list[dict] = []
        self.current_rec_idx = 0
        self.is_recording = False
        self.recording_frames: list[np.ndarray] = []
        self.recording_start_time = 0.0

        self.ref_cap: cv2.VideoCapture | None = None
        self.ref_frame: np.ndarray | None = None

        self.window_name = "OeGS Gebaerden-Erkennung"
        self.cap: cv2.VideoCapture | None = None

        self.is_predict_recording = False
        self.predict_recording_frames: list[np.ndarray] = []
        self.predict_recording_start_time = 0.0
        self.predict_result_text = ""

        self.video_button_rect = (0, 0, 0, 0)

    def run(self) -> None:
        """Main entry point."""
        ensure_dirs()

        if not self.sign_list:
            print("[FEHLER] Keine Gebaerden in gebaerden.txt!")
            return

        print(f"\nGeladene Gebaerden: {len(self.sign_list)}")
        for s in self.sign_list:
            print(f"  - {s}")

        if MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            print("\nTrainiertes Modell gefunden. Lade Modell...")
            self._load_model()
            self.pipeline_done = True
        else:
            print("\nKein Modell gefunden. Starte Trainingspipeline...")
            pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
            pipeline_thread.start()

        print("\nInitialisiere MediaPipe...")
        self.detector = HolisticDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            print("[FEHLER] Webcam konnte nicht geoeffnet werden!")
            return

        print("Webcam geoeffnet. Druecke Q zum Beenden.\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1100, 620)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.frame_count += 1

            if self.detector:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_result = self.detector.process(rgb)

            if (
                self.clf is not None
                and self.last_result
                and not self.recording_mode
                and not self.is_predict_recording
            ):
                has_hands = bool(
                    self.last_result.left_hand_landmarks
                    or self.last_result.right_hand_landmarks
                )
                if has_hands:
                    landmarks = extract_landmarks_from_result(self.last_result)
                    self.landmark_buffer.append(landmarks.astype(np.float32))
                    if self.frame_count % 5 == 0 and len(self.landmark_buffer) >= 10:
                        self._predict()
                else:
                    self.landmark_buffer.clear()
                    self.current_prediction = ""
                    self.confidence = 0.0

            display = frame.copy()

            if self.last_result:
                _draw_landmarks(display, self.last_result)

            if self.recording_mode and not self.is_recording:
                display = self._build_recording_view(display)
            elif self.recording_mode and self.is_recording:
                self._handle_recording(frame)
                display = self._build_recording_progress(display)
            elif self.is_predict_recording:
                self._handle_predict_recording(frame)
                display = self._build_predict_recording_progress(display)
            else:
                display = self._build_recognition_view(display)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and self.recording_mode and not self.is_recording:
                self._start_recording()
            elif key == ord("n") and self.recording_mode and not self.is_recording:
                self._next_sign()
            elif key == ord("p") and self.recording_mode and not self.is_recording:
                self._prev_sign()
            elif key == ord("m") and not self.is_predict_recording:
                self._toggle_mode()
            elif key == ord("v") and not self.recording_mode and not self.is_predict_recording:
                self._start_predict_recording()

        self.cap.release()
        self._close_ref_video()
        cv2.destroyAllWindows()
        if self.detector:
            self.detector.close()

    def _run_pipeline(self) -> None:
        """Run download -> extract -> train pipeline automatically."""
        import shutil
        from src.config import LANDMARKS_DIR
        from src.scraper import download_all_signs

        if LANDMARKS_DIR.exists():
            shutil.rmtree(LANDMARKS_DIR)
            LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
            print("Alte Landmarks geloescht (Feature-Format aktualisiert)")

        self.pipeline_status = "Videos herunterladen..."
        print("\n=== Videos herunterladen ===")
        download_all_signs(self.sign_list)

        total_videos = 0
        signs_with_enough = 0
        for sign_name in self.sign_list:
            count = get_video_count(sign_name) + get_recording_count(sign_name)
            total_videos += count
            if count >= MIN_VIDEOS_PER_SIGN:
                signs_with_enough += 1
            print(f"  {sign_name}: {count} Videos")

        print(f"\n  {signs_with_enough}/{len(self.sign_list)} Gebaerden haben genug Videos")

        self.pipeline_status = "Landmarks extrahieren..."
        print("\n=== Landmarks extrahieren ===")
        from src.landmark_extractor import extract_all_landmarks
        extract_all_landmarks(self.sign_list)

        self.pipeline_status = "Modell trainieren..."
        print("\n=== Modell trainieren ===")
        from src.trainer import train_model
        success = train_model(self.sign_list)

        if success:
            self._load_model()
            self.pipeline_status = "Bereit!"
            print("\n>>> Modell erfolgreich geladen! Erkennung aktiv. <<<")
        else:
            self.pipeline_status = "Training fehlgeschlagen"
            print("\n>>> Training fehlgeschlagen <<<")

        self._check_signs_needing_data()

        self.pipeline_done = True
        print("\n=== Pipeline abgeschlossen ===")
        if self.clf is not None:
            print(f">>> Modell bereit mit {len(self.le.classes_)} Gebaerden <<<")
        else:
            print(">>> Kein Modell - Aufnahme-Modus aktiv <<<")

    def _load_model(self) -> None:
        """Load the trained model."""
        if not MODEL_PATH.exists():
            print(f"  Modell-Datei nicht gefunden: {MODEL_PATH}")
            return
        from src.trainer import load_model
        result = load_model()
        if result:
            self.clf, self.le = result
            print(f"  Modell geladen: {len(self.le.classes_)} Gebaerden")

    def _check_signs_needing_data(self) -> None:
        """Check which signs need more recordings."""
        self.signs_needing_data = []
        for sign_name in self.sign_list:
            videos = get_video_count(sign_name)
            recordings = get_recording_count(sign_name)
            total = videos + recordings
            if total < MIN_VIDEOS_PER_SIGN:
                self.signs_needing_data.append({
                    "name": sign_name,
                    "total": total,
                    "needed": MIN_VIDEOS_PER_SIGN - total,
                })

        if self.signs_needing_data and self.clf is None:
            self.recording_mode = True
            self.current_rec_idx = 0
            self._open_reference_video()

    def _open_reference_video(self) -> None:
        """Open the reference video for the current sign as a VideoCapture."""
        self._close_ref_video()
        self.ref_frame = None
        if not self.signs_needing_data:
            return
        sign = self.signs_needing_data[self.current_rec_idx]
        safe_name = _sanitize_dirname(sign["name"])
        video_dir = VIDEOS_DIR / safe_name
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            if videos:
                self.ref_cap = cv2.VideoCapture(str(videos[0]))

    def _close_ref_video(self) -> None:
        """Release the reference video capture."""
        if self.ref_cap is not None:
            self.ref_cap.release()
            self.ref_cap = None

    def _read_ref_frame(self) -> np.ndarray | None:
        """Read the next frame from the reference video (loops)."""
        if self.ref_cap is None or not self.ref_cap.isOpened():
            return self.ref_frame
        ret, frame = self.ref_cap.read()
        if not ret:
            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.ref_cap.read()
        if ret:
            self.ref_frame = frame
        return self.ref_frame

    def _predict(self) -> None:
        """Predict the current sign from the buffer."""
        if self.clf is None or self.le is None:
            return

        from src.landmark_extractor import pad_or_truncate_sequence
        from src.trainer import predict_with_features

        padded = pad_or_truncate_sequence(
            list(self.landmark_buffer),
            LANDMARK_SEQUENCE_LENGTH,
        )
        proba = predict_with_features(self.clf, padded)[0]
        max_idx = int(np.argmax(proba))
        conf = float(proba[max_idx])

        threshold = 0.5
        if conf >= threshold:
            pred = self.le.inverse_transform([max_idx])[0]
            self.current_prediction = pred
        else:
            self.current_prediction = ""
        self.confidence = conf

    def _predict_from_frames(self, frames: list[np.ndarray]) -> tuple[str, float]:
        """Predict a sign from recorded frames."""
        if self.clf is None or self.le is None or not frames:
            return "", 0.0

        from src.landmark_extractor import pad_or_truncate_sequence
        from src.trainer import predict_with_features

        sequence: list[np.ndarray] = []

        with HolisticDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as detector:
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)

                has_hands = bool(
                    result.left_hand_landmarks or result.right_hand_landmarks
                )
                if not has_hands:
                    continue

                landmarks = extract_landmarks_from_result(result)
                sequence.append(landmarks.astype(np.float32))

        if not sequence:
            return "", 0.0

        padded = pad_or_truncate_sequence(sequence, LANDMARK_SEQUENCE_LENGTH)
        proba = predict_with_features(self.clf, padded)[0]
        max_idx = int(np.argmax(proba))
        conf = float(proba[max_idx])
        pred = self.le.inverse_transform([max_idx])[0]

        if conf < 0.4:
            return "", conf
        return pred, conf

    def _build_recognition_view(self, frame: np.ndarray) -> np.ndarray:
        """Build the recognition overlay on the frame."""
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 40), -1)
        cv2.putText(
            frame,
            "OeGS Gebaerden-Erkennung",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if not self.pipeline_done:
            cv2.putText(
                frame,
                self.pipeline_status,
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (100, 200, 255),
                1,
            )
        elif self.clf is None:
            cv2.putText(
                frame,
                "Kein Modell - druecke M fuer Aufnahme-Modus",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 100, 255),
                1,
            )
        else:
            cv2.putText(
                frame,
                "Erkennung aktiv - zeige eine Gebaerde",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (100, 255, 100),
                1,
            )

        if self.current_prediction and self.clf is not None:
            cv2.rectangle(frame, (0, h - 95), (w, h), (20, 20, 40), -1)

            if self.confidence >= 0.6:
                pred_color = (100, 255, 100)
                label = self.current_prediction
            elif self.confidence >= 0.4:
                pred_color = (100, 220, 255)
                label = f"{self.current_prediction} (?)"
            else:
                pred_color = (100, 100, 200)
                label = f"Unsicher: {self.current_prediction}"

            cv2.putText(
                frame,
                label,
                (15, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                pred_color,
                3,
            )
            conf_text = f"Konfidenz: {self.confidence:.0%}"
            cv2.putText(
                frame,
                conf_text,
                (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            bar_w = int((w - 30) * self.confidence)
            cv2.rectangle(frame, (15, h - 100), (w - 15, h - 96), (40, 40, 60), -1)
            cv2.rectangle(frame, (15, h - 100), (15 + bar_w, h - 96), pred_color, -1)

        if self.pipeline_done and self.clf is not None:
            x1, y1, x2, y2 = w - 245, 18, w - 20, 54
            self.video_button_rect = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 120, 220), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 255), 2)
            cv2.putText(
                frame,
                "Video aufnehmen",
                (x1 + 14, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Q=Beenden  M=Aufnahme-Modus  V=Video",
                (w - 360, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )
        else:
            self.video_button_rect = (0, 0, 0, 0)

        if self.predict_result_text:
            cv2.rectangle(frame, (15, 90), (min(w - 15, 620), 130), (25, 25, 55), -1)
            cv2.putText(
                frame,
                self.predict_result_text,
                (25, 117),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 220, 255),
                1,
            )

        return frame

    def _build_recording_view(self, frame: np.ndarray) -> np.ndarray:
        """Build the recording mode view with reference video side by side."""
        h, w = frame.shape[:2]

        if not self.signs_needing_data:
            cv2.putText(
                frame,
                "Alle Gebaerden haben genug Daten!",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 255, 100),
                2,
            )
            return frame

        sign = self.signs_needing_data[self.current_rec_idx]
        ref_frame = self._read_ref_frame()

        if ref_frame is not None:
            half_w = w // 2
            cam_view = _resize_keep_aspect(frame, half_w, h)
            ref_view = _resize_keep_aspect(ref_frame, half_w, h)
            combined = np.hstack([cam_view, ref_view])
        else:
            combined = frame.copy()
            cv2.rectangle(combined, (w // 2, 0), (w, 70), (40, 20, 20), -1)
            cv2.putText(
                combined,
                "Kein Referenzvideo gefunden!",
                (w // 2 + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 255),
                2,
            )

        ch, cw = combined.shape[:2]
        mid = cw // 2 if ref_frame is not None else cw

        cv2.rectangle(combined, (0, 0), (mid, 70), (20, 20, 40), -1)
        cv2.putText(
            combined,
            f"AUFNAHME: {sign['name']}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 255),
            2,
        )
        cv2.putText(
            combined,
            f"Vorhanden: {sign['total']}/{MIN_VIDEOS_PER_SIGN} (noch {sign['needed']} noetig)",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 200, 255),
            1,
        )

        if ref_frame is not None:
            cv2.rectangle(combined, (mid, 0), (cw, 40), (20, 20, 40), -1)
            cv2.putText(
                combined,
                "Referenzvideo",
                (mid + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

        cv2.rectangle(combined, (0, ch - 35), (cw, ch), (20, 20, 40), -1)
        nav = f"Gebaerde {self.current_rec_idx + 1}/{len(self.signs_needing_data)}"
        cv2.putText(
            combined,
            f"LEERTASTE=Aufnahme  N/P=Wechseln  M=Erkennung  Q=Beenden  |  {nav}",
            (10, ch - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        return combined

    def _build_recording_progress(self, frame: np.ndarray) -> np.ndarray:
        """Build the view during active training recording."""
        h, w = frame.shape[:2]
        elapsed = time.time() - self.recording_start_time
        progress = min(1.0, elapsed / RECORDING_DURATION_SECONDS)

        if self.last_result:
            _draw_landmarks(frame, self.last_result)

        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 60), -1)
        cv2.circle(frame, (25, 25), 10, (0, 0, 255), -1)
        sign = self.signs_needing_data[self.current_rec_idx]
        cv2.putText(
            frame,
            f"AUFNAHME DATEN: {sign['name']} ({progress:.0%})",
            (45, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        bar_w = int((w - 20) * progress)
        cv2.rectangle(frame, (10, h - 15), (w - 10, h - 5), (40, 40, 60), -1)
        cv2.rectangle(frame, (10, h - 15), (10 + bar_w, h - 5), (0, 0, 255), -1)

        return frame

    def _build_predict_recording_progress(self, frame: np.ndarray) -> np.ndarray:
        """Build the view during active prediction recording."""
        h, w = frame.shape[:2]
        elapsed = time.time() - self.predict_recording_start_time
        progress = min(1.0, elapsed / RECORDING_DURATION_SECONDS)

        cv2.rectangle(frame, (0, 0), (w, 60), (50, 20, 20), -1)
        cv2.circle(frame, (28, 30), 11, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"VIDEO-AUFNAHME fuer Erkennung ({progress:.0%})",
            (52, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Zeige jetzt die Gebaerde in die Kamera",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 220, 255),
            2,
        )

        bar_w = int((w - 20) * progress)
        cv2.rectangle(frame, (10, h - 18), (w - 10, h - 8), (40, 40, 60), -1)
        cv2.rectangle(frame, (10, h - 18), (10 + bar_w, h - 8), (0, 0, 255), -1)

        return frame

    def _start_recording(self) -> None:
        """Start recording training data for a sign."""
        if not self.signs_needing_data or self.cap is None:
            return

        sign = self.signs_needing_data[self.current_rec_idx]
        print(f"\nAufnahme fuer '{sign['name']}' startet...")

        for i in range(3, 0, -1):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                cv2.putText(
                    frame,
                    str(i),
                    (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5.0,
                    (247, 195, 79),
                    8,
                )
                cv2.putText(
                    frame,
                    f"Gebaerde: {sign['name']}",
                    (w // 2 - 120, h // 2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                )
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1000)

        self.is_recording = True
        self.recording_frames = []
        self.recording_start_time = time.time()

    def _handle_recording(self, frame: np.ndarray) -> None:
        """Handle frame during training recording."""
        self.recording_frames.append(frame.copy())
        elapsed = time.time() - self.recording_start_time
        if elapsed >= RECORDING_DURATION_SECONDS:
            self._save_recording()

    def _save_recording(self) -> None:
        """Save the recorded frames as mp4."""
        self.is_recording = False
        sign = self.signs_needing_data[self.current_rec_idx]
        safe_name = _sanitize_dirname(sign["name"])
        sign_dir = RECORDINGS_DIR / safe_name
        sign_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        save_path = sign_dir / f"recording_{timestamp}.mp4"

        if self.recording_frames:
            h, w = self.recording_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(save_path), fourcc, RECORDING_FPS, (w, h))
            for f in self.recording_frames:
                out.write(f)
            out.release()

        self.recording_frames = []
        print(f"  Aufnahme gespeichert: {save_path}")

        sign["total"] = get_video_count(sign["name"]) + get_recording_count(sign["name"])
        sign["needed"] = max(0, MIN_VIDEOS_PER_SIGN - sign["total"])

        if sign["needed"] == 0:
            print(f"  '{sign['name']}' hat jetzt genug Daten!")

    def _start_predict_recording(self) -> None:
        """Start a short recording for direct prediction."""
        if self.cap is None or self.clf is None or self.le is None or self.recording_mode:
            return

        self.predict_result_text = ""

        for i in range(3, 0, -1):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                cv2.putText(
                    frame,
                    str(i),
                    (w // 2 - 40, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5.0,
                    (80, 200, 255),
                    8,
                )
                cv2.putText(
                    frame,
                    "Videoaufnahme fuer Erkennung",
                    (w // 2 - 210, h // 2 + 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (230, 230, 230),
                    2,
                )
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1000)

        self.is_predict_recording = True
        self.predict_recording_frames = []
        self.predict_recording_start_time = time.time()

    def _handle_predict_recording(self, frame: np.ndarray) -> None:
        """Handle frame during prediction recording."""
        self.predict_recording_frames.append(frame.copy())
        elapsed = time.time() - self.predict_recording_start_time
        if elapsed >= RECORDING_DURATION_SECONDS:
            self._finish_predict_recording()

    def _finish_predict_recording(self) -> None:
        """Finish prediction recording, save it, analyze it, and show popup."""
        self.is_predict_recording = False

        if not self.predict_recording_frames:
            self._show_popup("Erkennung", "Keine Frames aufgenommen.")
            return

        predict_dir = RECORDINGS_DIR / "_predictions"
        predict_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        save_path = predict_dir / f"prediction_{timestamp}.mp4"

        h, w = self.predict_recording_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(save_path), fourcc, RECORDING_FPS, (w, h))
        for f in self.predict_recording_frames:
            out.write(f)
        out.release()

        pred, conf = self._predict_from_frames(self.predict_recording_frames)
        self.predict_recording_frames = []

        if not pred:
            self.current_prediction = ""
            self.confidence = conf
            self.predict_result_text = "Keine sichere Erkennung"
            self._show_popup(
                "Erkennung",
                "Keine sichere Gebaerde erkannt.",
            )
            return

        self.current_prediction = pred
        self.confidence = conf
        self.predict_result_text = f"Letztes Video: {pred} ({conf:.0%})"

        self._show_popup(
            "Erkennung",
            f"Erkannte Gebaerde: {pred}\nKonfidenz: {conf:.0%}\nGespeichert unter:\n{save_path}",
        )

    def _show_popup(self, title: str, message: str) -> None:
        """Show a popup message if possible."""
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showinfo(title, message)
            root.destroy()
        except Exception:
            print(f"[POPUP] {title}: {message}")

    def _next_sign(self) -> None:
        """Move to the next sign needing data."""
        if self.signs_needing_data:
            self.current_rec_idx = min(
                self.current_rec_idx + 1,
                len(self.signs_needing_data) - 1,
            )
            self._open_reference_video()

    def _prev_sign(self) -> None:
        """Move to the previous sign needing data."""
        self.current_rec_idx = max(self.current_rec_idx - 1, 0)
        self._open_reference_video()

    def _toggle_mode(self) -> None:
        """Toggle between recognition and recording mode."""
        if self.recording_mode:
            self.recording_mode = False
            self._close_ref_video()
            self._load_model()
        else:
            self._check_signs_needing_data()
            if not self.signs_needing_data:
                print("Alle Gebaerden haben genug Daten!")
            self.recording_mode = True

    def _on_mouse(self, event, x, y, flags, param) -> None:
        """Handle button clicks inside the OpenCV window."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.recording_mode or self.is_recording or self.is_predict_recording:
            return

        x1, y1, x2, y2 = self.video_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._start_predict_recording()


def run_desktop_app() -> None:
    """Entry point for the desktop application."""
    app = DesktopApp()
    app.run()