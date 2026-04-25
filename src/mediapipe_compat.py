"""MediaPipe compatibility layer for old (solutions) and new (tasks) API."""

import urllib.request

import mediapipe as mp
import numpy as np

from src.config import MODELS_DIR

_USE_LEGACY = hasattr(mp, "solutions")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)
_MODEL_PATH = MODELS_DIR / "holistic_landmarker.task"


class HolisticResult:
    """Unified result from holistic landmark detection."""

    def __init__(
        self,
        left_hand_landmarks: list | None,
        right_hand_landmarks: list | None,
        pose_landmarks: list | None,
    ):
        self.left_hand_landmarks = left_hand_landmarks
        self.right_hand_landmarks = right_hand_landmarks
        self.pose_landmarks = pose_landmarks


def _download_model() -> None:
    """Download the holistic landmarker model if not present."""
    if _MODEL_PATH.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Lade Holistic-Landmarker-Modell herunter...")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"Modell gespeichert: {_MODEL_PATH}")


class HolisticDetector:
    """Wrapper providing a unified interface for both mediapipe API versions."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._use_legacy = _USE_LEGACY

        if self._use_legacy:
            mp_holistic = mp.solutions.holistic
            self._holistic = mp_holistic.Holistic(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            _download_model()
            options = mp.tasks.vision.HolisticLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(_MODEL_PATH)
                ),
                min_hand_landmarks_confidence=min_detection_confidence,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_landmarks_confidence=min_tracking_confidence,
            )
            self._landmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(
                options
            )

    def process(self, rgb_frame: np.ndarray) -> HolisticResult:
        """Process an RGB frame and return unified landmarks."""
        if self._use_legacy:
            rgb_frame.flags.writeable = False
            results = self._holistic.process(rgb_frame)
            return HolisticResult(
                left_hand_landmarks=results.left_hand_landmarks,
                right_hand_landmarks=results.right_hand_landmarks,
                pose_landmarks=results.pose_landmarks,
            )
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self._landmarker.detect(mp_image)
            return HolisticResult(
                left_hand_landmarks=(
                    results.left_hand_landmarks[0]
                    if results.left_hand_landmarks
                    else None
                ),
                right_hand_landmarks=(
                    results.right_hand_landmarks[0]
                    if results.right_hand_landmarks
                    else None
                ),
                pose_landmarks=(
                    results.pose_landmarks[0]
                    if results.pose_landmarks
                    else None
                ),
            )

    def close(self) -> None:
        """Release resources."""
        if self._use_legacy:
            self._holistic.close()
        else:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def extract_landmarks_from_result(result: HolisticResult) -> np.ndarray:
    """Extract a flat landmark array from a HolisticResult.

    Returns a flattened numpy array:
    - Left hand: 21 landmarks * 3 (x,y,z) = 63
    - Right hand: 21 landmarks * 3 (x,y,z) = 63
    - Pose: 33 landmarks * 4 (x,y,z,visibility) = 132
    Total: 258 values
    """
    landmarks: list[float] = []

    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    if result.pose_landmarks:
        for lm in result.pose_landmarks:
            vis = getattr(lm, "visibility", 0.0) or 0.0
            landmarks.extend([lm.x, lm.y, lm.z, vis])
    else:
        landmarks.extend([0.0] * 132)

    return np.array(landmarks)


def draw_landmarks_on_frame(frame: np.ndarray, result: HolisticResult) -> None:
    """Draw landmarks on a BGR frame (only available with legacy API)."""
    if not _USE_LEGACY:
        return
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    class _FakeResults:
        pass

    fake = _FakeResults()
    fake.left_hand_landmarks = result.left_hand_landmarks
    fake.right_hand_landmarks = result.right_hand_landmarks
    fake.pose_landmarks = result.pose_landmarks

    if fake.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            fake.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    if fake.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            fake.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    if fake.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            fake.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
