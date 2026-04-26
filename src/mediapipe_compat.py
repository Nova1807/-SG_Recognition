"""MediaPipe compatibility layer for old (solutions) and new (tasks) API."""

import urllib.request

import mediapipe as mp
import numpy as np

from src.config import (
    FEATURES_PER_FRAME,
    HAND_FEATURES_PER_HAND,
    LEFT_HAND_END,
    LEFT_HAND_START,
    LEFT_PRESENT_IDX,
    MODELS_DIR,
    PRIMARY_HAND_END,
    PRIMARY_HAND_START,
    PRIMARY_PRESENT_IDX,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
    RIGHT_PRESENT_IDX,
)

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
    if _MODEL_PATH.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Lade Holistic-Landmarker-Modell herunter...")
    urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    print(f"Modell gespeichert: {_MODEL_PATH}")


def _flatten_landmarks(lm_data):
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
        if self._use_legacy:
            rgb_frame.flags.writeable = False
            results = self._holistic.process(rgb_frame)
            return HolisticResult(
                left_hand_landmarks=results.left_hand_landmarks,
                right_hand_landmarks=results.right_hand_landmarks,
                pose_landmarks=results.pose_landmarks,
            )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._landmarker.detect(mp_image)
        return HolisticResult(
            left_hand_landmarks=_flatten_landmarks(results.left_hand_landmarks),
            right_hand_landmarks=_flatten_landmarks(results.right_hand_landmarks),
            pose_landmarks=_flatten_landmarks(results.pose_landmarks),
        )

    def close(self) -> None:
        if self._use_legacy:
            self._holistic.close()
        else:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _as_xyz_points(hand_landmarks) -> np.ndarray | None:
    if not hand_landmarks:
        return None

    first = hand_landmarks[0]
    if isinstance(first, tuple) and len(first) == 3:
        arr = np.array(hand_landmarks, dtype=np.float32)
    else:
        arr = np.array(
            [(float(lm.x), float(lm.y), float(lm.z)) for lm in hand_landmarks],
            dtype=np.float32,
        )

    if arr.shape != (21, 3):
        return None
    return arr


def _normalize_hand_landmarks(
    hand_landmarks, mirror_x: bool
) -> np.ndarray | None:
    pts = _as_xyz_points(hand_landmarks)
    if pts is None:
        return None

    pts = pts.copy()
    if mirror_x:
        pts[:, 0] = 1.0 - pts[:, 0]

    wrist = pts[0].copy()
    pts = pts - wrist

    ref = float(np.linalg.norm(pts[9]))
    if ref < 1e-6:
        ref = float(np.linalg.norm(pts[5]))
    if ref < 1e-6:
        ref = float(np.linalg.norm(pts[17]))
    if ref < 1e-6:
        ref = 1.0

    pts = pts / ref
    return pts.astype(np.float32)


def _hand_salience(hand_landmarks) -> float:
    pts = _as_xyz_points(hand_landmarks)
    if pts is None:
        return 0.0

    span = pts.max(axis=0) - pts.min(axis=0)
    tip_reach = np.linalg.norm(pts[[4, 8, 12, 16, 20]] - pts[0], axis=1).mean()
    return float((span[0] * span[1]) + 0.35 * tip_reach)


def _flatten_or_zero(hand_array: np.ndarray | None) -> np.ndarray:
    if hand_array is None:
        return np.zeros(HAND_FEATURES_PER_HAND, dtype=np.float32)
    return hand_array.reshape(-1).astype(np.float32)


def _select_primary_hand(
    left_raw,
    right_raw,
    left_norm: np.ndarray | None,
    right_norm: np.ndarray | None,
) -> np.ndarray | None:
    if left_raw and not right_raw:
        return left_norm
    if right_raw and not left_raw:
        return right_norm
    if not left_raw and not right_raw:
        return None

    left_score = _hand_salience(left_raw)
    right_score = _hand_salience(right_raw)

    if left_score > right_score * 1.10:
        return left_norm
    if right_score > left_score * 1.10:
        return right_norm

    left_extent = 0.0 if left_norm is None else float(np.abs(left_norm).mean())
    right_extent = 0.0 if right_norm is None else float(np.abs(right_norm).mean())

    if left_extent > right_extent:
        return left_norm
    return right_norm


def extract_landmarks_from_result(result: HolisticResult) -> np.ndarray:
    """Feature layout per frame:
    [primary_present, left_present, right_present,
     primary_hand(63), left_hand(63), right_hand(63)]

    - primary_hand: canonical active/dominant hand, left mirrored to right-like form
    - left_hand: physical left hand, mirrored to right-like canonical form
    - right_hand: physical right hand, kept as right-like canonical form
    """
    left_norm = _normalize_hand_landmarks(
        result.left_hand_landmarks, mirror_x=True
    )
    right_norm = _normalize_hand_landmarks(
        result.right_hand_landmarks, mirror_x=False
    )
    primary_norm = _select_primary_hand(
        result.left_hand_landmarks,
        result.right_hand_landmarks,
        left_norm,
        right_norm,
    )

    features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

    if primary_norm is not None:
        features[PRIMARY_PRESENT_IDX] = 1.0
        features[PRIMARY_HAND_START:PRIMARY_HAND_END] = _flatten_or_zero(
            primary_norm
        )

    if left_norm is not None:
        features[LEFT_PRESENT_IDX] = 1.0
        features[LEFT_HAND_START:LEFT_HAND_END] = _flatten_or_zero(left_norm)

    if right_norm is not None:
        features[RIGHT_PRESENT_IDX] = 1.0
        features[RIGHT_HAND_START:RIGHT_HAND_END] = _flatten_or_zero(right_norm)

    return features


def draw_landmarks_on_frame(frame: np.ndarray, result: HolisticResult) -> None:
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