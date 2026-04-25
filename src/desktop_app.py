"""Desktop popup application for ÖGS sign recognition."""

import threading
import time
from collections import deque

import cv2
import numpy as np

from src.config import (
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
from src.landmark_extractor import extract_landmarks_from_frame
from src.mediapipe_compat import HolisticDetector
from src.scraper import (
    _sanitize_dirname,
    get_recording_count,
    get_video_count,
)


class DesktopApp:
    """Main desktop application for sign recognition."""

    def __init__(self):
        self.sign_list = load_sign_list()
        self.detector = None
        self.clf = None
        self.le = None
        self.pipeline_done = False
        self.pipeline_status = "Starte..."
        self.landmark_buffer: deque[np.ndarray] = deque(
            maxlen=LANDMARK_SEQUENCE_LENGTH
        )
        self.current_prediction = ""
        self.confidence = 0.0
        self.is_hands_active = False
        self.history: list[tuple[str, float]] = []

        # Recording state
        self.recording_mode = False
        self.signs_needing_data: list[dict] = []
        self.current_rec_idx = 0
        self.is_recording = False
        self.recording_countdown = 0
        self.recording_frames: list[np.ndarray] = []
        self.recording_start_time = 0.0
        self.reference_frame: np.ndarray | None = None

    def run(self) -> None:
        """Main entry point."""
        ensure_dirs()

        if not self.sign_list:
            print("[FEHLER] Keine Gebärden in gebaerden.txt!")
            return

        print(f"\nGeladene Gebärden: {len(self.sign_list)}")
        for s in self.sign_list:
            print(f"  - {s}")

        # Run pipeline in background
        pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        pipeline_thread.start()

        # Initialize detector
        print("\nInitialisiere MediaPipe...")
        self.detector = HolisticDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Open webcam
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print("[FEHLER] Webcam konnte nicht geöffnet werden!")
            return

        print("Webcam geöffnet. Drücke Q zum Beenden.\n")

        window_name = "OeGS Gebaerden-Erkennung"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            if self.pipeline_done and self.clf is not None:
                self._process_recognition(frame)

            if self.recording_mode and not self.is_recording:
                display = self._build_recording_view(display)
            elif self.recording_mode and self.is_recording:
                self._handle_recording(frame)
                display = self._build_recording_progress(display)
            else:
                display = self._build_recognition_view(display)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and self.recording_mode and not self.is_recording:
                self._start_recording(cap)
            elif key == ord("n") and self.recording_mode and not self.is_recording:
                self._next_sign()
            elif key == ord("p") and self.recording_mode and not self.is_recording:
                self._prev_sign()
            elif key == ord("m"):
                self._toggle_mode()

        cap.release()
        cv2.destroyAllWindows()
        if self.detector:
            self.detector.close()

    def _run_pipeline(self) -> None:
        """Run download → extract → train pipeline automatically."""
        from src.scraper import download_all_signs

        # Download
        self.pipeline_status = "Videos herunterladen..."
        print("\n=== Videos herunterladen ===")
        download_all_signs(self.sign_list)

        # Extract landmarks
        self.pipeline_status = "Landmarks extrahieren..."
        print("\n=== Landmarks extrahieren ===")
        from src.landmark_extractor import extract_all_landmarks
        extract_all_landmarks(self.sign_list)

        # Train
        self.pipeline_status = "Modell trainieren..."
        print("\n=== Modell trainieren ===")
        from src.trainer import train_model
        success = train_model(self.sign_list)

        if success:
            self._load_model()
            self.pipeline_status = "Bereit!"
        else:
            self.pipeline_status = "Zu wenig Daten - Aufnahme nötig"
            self._check_signs_needing_data()

        self.pipeline_done = True
        print("\n=== Pipeline abgeschlossen ===\n")

    def _load_model(self) -> None:
        """Load the trained model."""
        if not MODEL_PATH.exists():
            return
        from src.trainer import load_model
        result = load_model()
        if result:
            self.clf, self.le = result

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

        if self.signs_needing_data:
            self.recording_mode = True
            self.current_rec_idx = 0
            self._load_reference_video()

    def _load_reference_video(self) -> None:
        """Load a reference video frame for the current sign."""
        self.reference_frame = None
        if not self.signs_needing_data:
            return
        sign = self.signs_needing_data[self.current_rec_idx]
        safe_name = _sanitize_dirname(sign["name"])
        video_dir = VIDEOS_DIR / safe_name
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            if videos:
                cap = cv2.VideoCapture(str(videos[0]))
                ret, frame = cap.read()
                if ret:
                    self.reference_frame = frame
                cap.release()

    def _process_recognition(self, frame: np.ndarray) -> None:
        """Process a frame for sign recognition."""
        landmarks = extract_landmarks_from_frame(frame, self.detector)
        if landmarks is None:
            return

        hand_data = landmarks[:126]
        has_hands = bool(np.any(hand_data != 0))

        if has_hands:
            if not self.is_hands_active:
                self.is_hands_active = True
                self.landmark_buffer.clear()
            self.landmark_buffer.append(landmarks)
        else:
            if self.is_hands_active and len(self.landmark_buffer) >= 5:
                self._predict()
            self.is_hands_active = False

    def _predict(self) -> None:
        """Predict the current sign from the buffer."""
        if self.clf is None or self.le is None:
            return
        from src.landmark_extractor import pad_or_truncate_sequence

        padded = pad_or_truncate_sequence(
            list(self.landmark_buffer), LANDMARK_SEQUENCE_LENGTH
        )
        flat = padded.flatten().reshape(1, -1)
        proba = self.clf.predict_proba(flat)[0]
        max_idx = np.argmax(proba)
        conf = float(proba[max_idx])

        if conf >= 0.3:
            pred = self.le.inverse_transform([max_idx])[0]
            self.current_prediction = pred
            self.confidence = conf
            self.history.insert(0, (pred, conf))
            if len(self.history) > 10:
                self.history.pop()

    def _build_recognition_view(self, frame: np.ndarray) -> np.ndarray:
        """Build the recognition overlay on the frame."""
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 40), -1)
        cv2.putText(
            frame, "OeGS Gebaerden-Erkennung",
            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        if not self.pipeline_done:
            cv2.putText(
                frame, self.pipeline_status,
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1,
            )
        else:
            status = "Haende erkannt" if self.is_hands_active else "Warte auf Gebaerde..."
            color = (0, 100, 255) if self.is_hands_active else (100, 255, 100)
            cv2.putText(
                frame, status,
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        # Prediction box (bottom)
        if self.current_prediction:
            cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 40), -1)
            cv2.putText(
                frame, self.current_prediction,
                (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (247, 195, 79), 3,
            )
            conf_text = f"Konfidenz: {self.confidence:.0%}"
            cv2.putText(
                frame, conf_text,
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )
            # Confidence bar
            bar_w = int((w - 30) * self.confidence)
            cv2.rectangle(frame, (15, h - 95), (w - 15, h - 92), (40, 40, 60), -1)
            bar_color = (100, 255, 100) if self.confidence > 0.7 else (100, 200, 255)
            cv2.rectangle(frame, (15, h - 95), (15 + bar_w, h - 92), bar_color, -1)

        # Controls hint
        if self.pipeline_done:
            cv2.putText(
                frame, "Q=Beenden  M=Aufnahme-Modus",
                (w - 310, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
            )

        return frame

    def _build_recording_view(self, frame: np.ndarray) -> np.ndarray:
        """Build the recording mode view with reference video."""
        h, w = frame.shape[:2]

        if not self.signs_needing_data:
            cv2.putText(
                frame, "Alle Gebaerden haben genug Daten!",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2,
            )
            return frame

        sign = self.signs_needing_data[self.current_rec_idx]

        # If reference video exists, show side by side
        if self.reference_frame is not None:
            ref_resized = cv2.resize(self.reference_frame, (w // 2, h))
            cam_resized = cv2.resize(frame, (w // 2, h))
            combined = np.hstack([cam_resized, ref_resized])

            ch, cw = combined.shape[:2]

            # Reference label
            cv2.rectangle(combined, (cw // 2, 0), (cw, 40), (20, 20, 40), -1)
            cv2.putText(
                combined, "Referenzvideo",
                (cw // 2 + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )
        else:
            combined = frame.copy()
            ch, cw = combined.shape[:2]

        # Top bar
        cv2.rectangle(combined, (0, 0), (cw // 2 if self.reference_frame is not None else cw, 70), (20, 20, 40), -1)
        cv2.putText(
            combined, f"AUFNAHME: {sign['name']}",
            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2,
        )
        cv2.putText(
            combined,
            f"Vorhanden: {sign['total']}/{MIN_VIDEOS_PER_SIGN} (noch {sign['needed']} noetig)",
            (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1,
        )

        # Bottom controls
        cv2.rectangle(combined, (0, ch - 35), (cw, ch), (20, 20, 40), -1)
        nav = f"Gebaerde {self.current_rec_idx + 1}/{len(self.signs_needing_data)}"
        cv2.putText(
            combined, f"LEERTASTE=Aufnahme  N=Naechste  P=Vorherige  M=Erkennung  Q=Beenden  |  {nav}",
            (10, ch - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )

        return combined

    def _build_recording_progress(self, frame: np.ndarray) -> np.ndarray:
        """Build the view during active recording."""
        h, w = frame.shape[:2]
        elapsed = time.time() - self.recording_start_time
        progress = min(1.0, elapsed / RECORDING_DURATION_SECONDS)

        # Red recording indicator
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 60), -1)
        cv2.circle(frame, (25, 25), 10, (0, 0, 255), -1)
        sign = self.signs_needing_data[self.current_rec_idx]
        cv2.putText(
            frame, f"AUFNAHME: {sign['name']} ({progress:.0%})",
            (45, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )

        # Progress bar
        bar_w = int((w - 20) * progress)
        cv2.rectangle(frame, (10, h - 15), (w - 10, h - 5), (40, 40, 60), -1)
        cv2.rectangle(frame, (10, h - 15), (10 + bar_w, h - 5), (0, 0, 255), -1)

        return frame

    def _start_recording(self, cap: cv2.VideoCapture) -> None:
        """Start recording a sign."""
        if not self.signs_needing_data:
            return

        sign = self.signs_needing_data[self.current_rec_idx]
        print(f"\nAufnahme fuer '{sign['name']}' startet...")

        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.putText(
                    frame, str(i),
                    (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0, (247, 195, 79), 8,
                )
                cv2.putText(
                    frame, f"Gebaerde: {sign['name']}",
                    (w // 2 - 120, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
                )
                cv2.imshow("OeGS Gebaerden-Erkennung", frame)
                cv2.waitKey(1000)

        self.is_recording = True
        self.recording_frames = []
        self.recording_start_time = time.time()

    def _handle_recording(self, frame: np.ndarray) -> None:
        """Handle frame during recording."""
        elapsed = time.time() - self.recording_start_time
        self.recording_frames.append(frame.copy())

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
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(save_path), fourcc, RECORDING_FPS, (w, h)
            )
            for f in self.recording_frames:
                out.write(f)
            out.release()

        self.recording_frames = []
        print(f"  Aufnahme gespeichert: {save_path}")

        # Update counts
        sign["total"] = get_video_count(sign["name"]) + get_recording_count(sign["name"])
        sign["needed"] = max(0, MIN_VIDEOS_PER_SIGN - sign["total"])

        if sign["needed"] == 0:
            print(f"  '{sign['name']}' hat jetzt genug Daten!")

    def _next_sign(self) -> None:
        """Move to the next sign needing data."""
        if self.signs_needing_data:
            self.current_rec_idx = min(
                self.current_rec_idx + 1,
                len(self.signs_needing_data) - 1,
            )
            self._load_reference_video()

    def _prev_sign(self) -> None:
        """Move to the previous sign needing data."""
        self.current_rec_idx = max(self.current_rec_idx - 1, 0)
        self._load_reference_video()

    def _toggle_mode(self) -> None:
        """Toggle between recognition and recording mode."""
        if self.recording_mode:
            self.recording_mode = False
            self._load_model()
        else:
            self._check_signs_needing_data()
            if not self.signs_needing_data:
                print("Alle Gebaerden haben genug Daten!")
            self.recording_mode = True


def run_desktop_app() -> None:
    """Entry point for the desktop application."""
    app = DesktopApp()
    app.run()
