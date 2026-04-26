from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SIGN_LIST_FILE = BASE_DIR / "gebaerden.txt"

DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
RECORDINGS_DIR = DATA_DIR / "recordings"
LANDMARKS_DIR = DATA_DIR / "landmarks"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"

SEARCH_URL = "https://gebaerden-archiv.at/search"
MEDIA_BASE_URL = "https://gebaerden-archiv.at"

MIN_VIDEOS_PER_SIGN = 5
LANDMARK_SEQUENCE_LENGTH = 30

HAND_LANDMARKS_PER_HAND = 21
HAND_COORDS_PER_LANDMARK = 3
HAND_FEATURES_PER_HAND = HAND_LANDMARKS_PER_HAND * HAND_COORDS_PER_LANDMARK  # 63

PRIMARY_PRESENT_IDX = 0
LEFT_PRESENT_IDX = 1
RIGHT_PRESENT_IDX = 2

PRIMARY_HAND_START = 3
PRIMARY_HAND_END = PRIMARY_HAND_START + HAND_FEATURES_PER_HAND  # 66

LEFT_HAND_START = PRIMARY_HAND_END
LEFT_HAND_END = LEFT_HAND_START + HAND_FEATURES_PER_HAND  # 129

RIGHT_HAND_START = LEFT_HAND_END
RIGHT_HAND_END = RIGHT_HAND_START + HAND_FEATURES_PER_HAND  # 192

PRIMARY_WRIST_IDX = RIGHT_HAND_END          # 192
LEFT_WRIST_IDX = PRIMARY_WRIST_IDX + 2       # 194
RIGHT_WRIST_IDX = LEFT_WRIST_IDX + 2         # 196
FEATURES_PER_FRAME = RIGHT_WRIST_IDX + 2     # 198

RECORDING_FPS = 15
RECORDING_DURATION_SECONDS = 4

MODEL_PATH = MODELS_DIR / "sign_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
FEATURE_VERSION_PATH = MODELS_DIR / "feature_version.txt"

WEBCAM_INDEX = 0
UNKNOWN_SIGN_LABEL = "unknown"
REJECT_CONFIDENCE_THRESHOLD = 0.65
DISPLAY_CONFIDENCE_THRESHOLD = 0.45


def ensure_dirs() -> None:
    for d in [VIDEOS_DIR, RECORDINGS_DIR, LANDMARKS_DIR, UPLOADS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_sign_list() -> list[str]:
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