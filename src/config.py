"""Configuration for the ÖGS sign recognition program."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SIGN_LIST_FILE = BASE_DIR / "gebaerden.txt"

DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
RECORDINGS_DIR = DATA_DIR / "recordings"
LANDMARKS_DIR = DATA_DIR / "landmarks"
MODELS_DIR = BASE_DIR / "models"

SEARCH_URL = "https://gebaerden-archiv.at/search"
MEDIA_BASE_URL = "https://gebaerden-archiv.at"

MIN_VIDEOS_PER_SIGN = 5

LANDMARK_SEQUENCE_LENGTH = 30

NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33
HAND_DIMS = 3
POSE_DIMS = 4

FEATURES_PER_FRAME = (NUM_HAND_LANDMARKS * HAND_DIMS * 2) + (NUM_POSE_LANDMARKS * POSE_DIMS)

RECORDING_FPS = 15
RECORDING_DURATION_SECONDS = 4

MODEL_PATH = MODELS_DIR / "sign_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

WEBCAM_INDEX = 0


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in [VIDEOS_DIR, RECORDINGS_DIR, LANDMARKS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_sign_list() -> list[str]:
    """Load the list of accepted signs from the text file."""
    if not SIGN_LIST_FILE.exists():
        print(f"[FEHLER] Gebärden-Datei nicht gefunden: {SIGN_LIST_FILE}")
        return []

    signs = []
    with open(SIGN_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            sign = line.strip()
            if sign and not sign.startswith("#"):
                signs.append(sign)

    return signs
