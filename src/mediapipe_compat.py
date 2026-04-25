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


def _flatten_landmarks(lm_data):
    """Normalize landmark data to a flat list of landmark objects.

    The new tasks API may return landmarks as a flat list or empty list.
    The legacy API returns NormalizedLandmarkList objects (iterable).
    This ensures we always get a flat iterable of landmarks or None.
    """
    if not lm_data:
        return None
    if isinstance(lm_data, list) and len(lm_data) > 0:
        first = lm_data[0]
        if isinstance(first, list):
            return first
        if hasattr(first, "x"):
            return lm_data
    if hasattr(lm_data, "landmark"):
        return lm_data.landmark
    return lm_data


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
                left_hand_landmarks=_flatten_landmarks(
                    results.left_hand_landmarks
                ),
                right_hand_landmarks=_flatten_landmarks(
                    results.right_hand_landmarks
                ),
                pose_landmarks=_flatten_landmarks(
                    results.pose_landmarks
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


_FINGER_JOINTS = [
    # (MCP, PIP, DIP, TIP) landmark indices per finger
    (1, 2, 3, 4),      # Thumb
    (5, 6, 7, 8),      # Index
    (9, 10, 11, 12),    # Middle
    (13, 14, 15, 16),   # Ring
    (17, 18, 19, 20),   # Pinky
]


def _vec3(a: tuple, b: tuple) -> tuple:
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _dot3(a: tuple, b: tuple) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm3(v: tuple) -> float:
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def _angle_between(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at point b in radians (0 = straight, pi = fully bent)."""
    v1 = _vec3(b, a)
    v2 = _vec3(b, c)
    n1, n2 = _norm3(v1), _norm3(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_val = _dot3(v1, v2) / (n1 * n2)
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(np.arccos(cos_val))


def _extract_hand_features(hand_landmarks) -> list[float]:
    """Extract discriminative hand shape features.

    Per finger (5 fingers):
    - MCP angle (at base joint)
    - PIP angle (at middle joint)
    - DIP angle (at tip joint)
    - Fingertip distance to wrist (normalized)
    - Fingertip distance to palm center (normalized)
    Plus:
    - 10 inter-fingertip distances
    - Thumb-to-each-fingertip distances (4)

    Total: 5*5 + 10 + 4 = 39 features per hand
    """
    if not hand_landmarks:
        return [0.0] * 39

    pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    wrist = pts[0]
    palm_center = (
        (pts[0][0] + pts[5][0] + pts[9][0] + pts[13][0] + pts[17][0]) / 5,
        (pts[0][1] + pts[5][1] + pts[9][1] + pts[13][1] + pts[17][1]) / 5,
        (pts[0][2] + pts[5][2] + pts[9][2] + pts[13][2] + pts[17][2]) / 5,
    )

    # Normalization reference: wrist to middle finger MCP distance
    ref_dist = _norm3(_vec3(wrist, pts[9]))
    if ref_dist < 1e-8:
        ref_dist = 1.0

    features: list[float] = []

    tips = []
    for mcp, pip, dip, tip in _FINGER_JOINTS:
        # Joint angles
        mcp_angle = _angle_between(wrist, pts[mcp], pts[pip])
        pip_angle = _angle_between(pts[mcp], pts[pip], pts[dip])
        dip_angle = _angle_between(pts[pip], pts[dip], pts[tip])
        # Distances (normalized)
        tip_to_wrist = _norm3(_vec3(wrist, pts[tip])) / ref_dist
        tip_to_palm = _norm3(_vec3(palm_center, pts[tip])) / ref_dist

        features.extend([mcp_angle, pip_angle, dip_angle, tip_to_wrist, tip_to_palm])
        tips.append(pts[tip])

    # Inter-fingertip distances (10 pairs)
    for i in range(5):
        for j in range(i + 1, 5):
            dist = _norm3(_vec3(tips[i], tips[j])) / ref_dist
            features.append(dist)

    # Thumb tip to each other fingertip (4)
    for i in range(1, 5):
        dist = _norm3(_vec3(tips[0], tips[i])) / ref_dist
        features.append(dist)

    return features


# Features per hand: 39, two hands: 78
HAND_FEATURES_SIZE = 39
TOTAL_FEATURES_PER_FRAME = HAND_FEATURES_SIZE * 2  # 78 (hands only)


def extract_landmarks_from_result(result: HolisticResult) -> np.ndarray:
    """Extract discriminative hand shape features from a HolisticResult.

    Returns a flattened numpy array with angle and distance based features
    that are position-invariant, scale-invariant, and rotation-tolerant.

    - Left hand: 39 features (angles + distances)
    - Right hand: 39 features (angles + distances)
    Total: 78 values per frame
    """
    features: list[float] = []
    features.extend(_extract_hand_features(result.left_hand_landmarks))
    features.extend(_extract_hand_features(result.right_hand_landmarks))
    return np.array(features)


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
