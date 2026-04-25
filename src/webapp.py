"""Flask web application for ÖGS sign recognition."""

import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from src.config import (
    LANDMARKS_DIR,
    LANDMARK_SEQUENCE_LENGTH,
    MIN_VIDEOS_PER_SIGN,
    RECORDINGS_DIR,
    RECORDING_DURATION_SECONDS,
    VIDEOS_DIR,
    ensure_dirs,
    load_sign_list,
)
from src.landmark_extractor import extract_landmarks_from_frame, pad_or_truncate_sequence
from src.mediapipe_compat import HolisticDetector
from src.scraper import (
    _sanitize_dirname,
    get_recording_count,
    get_video_count,
    search_sign_videos,
)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "templates"),
    static_folder=str(Path(__file__).parent.parent / "static"),
)

download_progress: dict[str, object] = {
    "running": False,
    "current_sign": "",
    "total_signs": 0,
    "completed_signs": 0,
    "log": [],
}

extraction_progress: dict[str, object] = {
    "running": False,
    "current_sign": "",
    "total_signs": 0,
    "completed_signs": 0,
}

training_progress: dict[str, object] = {
    "running": False,
    "status": "",
    "accuracy": None,
}


@app.route("/")
def index():
    """Dashboard page."""
    sign_list = load_sign_list()
    signs_data = []
    for sign_name in sign_list:
        videos = get_video_count(sign_name)
        recordings = get_recording_count(sign_name)
        safe_name = _sanitize_dirname(sign_name)
        landmark_dir = LANDMARKS_DIR / safe_name
        landmarks = len(list(landmark_dir.glob("*.npy"))) if landmark_dir.exists() else 0
        total = videos + recordings

        signs_data.append({
            "name": sign_name,
            "videos": videos,
            "recordings": recordings,
            "landmarks": landmarks,
            "total": total,
            "needed": max(0, MIN_VIDEOS_PER_SIGN - total),
            "status": "ok" if total >= MIN_VIDEOS_PER_SIGN else ("wenig" if total > 0 else "fehlt"),
        })

    from src.config import MODEL_PATH
    model_exists = MODEL_PATH.exists()

    return render_template(
        "index.html",
        signs=signs_data,
        min_videos=MIN_VIDEOS_PER_SIGN,
        model_exists=model_exists,
    )


@app.route("/recognize")
def recognize_page():
    """Live recognition page."""
    return render_template("recognize.html")


@app.route("/record")
def record_page():
    """Recording page."""
    sign_list = load_sign_list()
    signs_data = []
    for sign_name in sign_list:
        videos = get_video_count(sign_name)
        recordings = get_recording_count(sign_name)
        total = videos + recordings
        needed = max(0, MIN_VIDEOS_PER_SIGN - total)

        reference_videos = []
        safe_name = _sanitize_dirname(sign_name)
        video_dir = VIDEOS_DIR / safe_name
        if video_dir.exists():
            for vf in list(video_dir.glob("*.mp4"))[:1]:
                reference_videos.append(f"/video/{safe_name}/{vf.name}")

        signs_data.append({
            "name": sign_name,
            "safe_name": safe_name,
            "videos": videos,
            "recordings": recordings,
            "total": total,
            "needed": needed,
            "reference_videos": reference_videos,
        })

    return render_template(
        "record.html",
        signs=signs_data,
        min_videos=MIN_VIDEOS_PER_SIGN,
        recording_duration=RECORDING_DURATION_SECONDS,
    )


@app.route("/video/<sign_name>/<filename>")
def serve_video(sign_name: str, filename: str):
    """Serve a video file."""
    video_path = VIDEOS_DIR / sign_name / filename
    if not video_path.exists():
        video_path = RECORDINGS_DIR / sign_name / filename
    if not video_path.exists():
        return "Video not found", 404

    def generate():
        with open(video_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk

    return Response(generate(), mimetype="video/mp4")


@app.route("/api/download", methods=["POST"])
def api_download():
    """Start video download in background."""
    if download_progress["running"]:
        return jsonify({"error": "Download läuft bereits"}), 400

    sign_list = load_sign_list()

    def run_download():
        download_progress["running"] = True
        download_progress["total_signs"] = len(sign_list)
        download_progress["completed_signs"] = 0
        download_progress["log"] = []

        for i, sign_name in enumerate(sign_list):
            download_progress["current_sign"] = sign_name
            download_progress["log"].append(f"Suche Videos für: {sign_name}...")

            results = search_sign_videos(sign_name)
            download_progress["log"].append(f"  {len(results)} Videos gefunden")

            safe_name = _sanitize_dirname(sign_name)
            sign_dir = VIDEOS_DIR / safe_name
            sign_dir.mkdir(parents=True, exist_ok=True)

            from src.scraper import download_video
            for result in results:
                filename = f"{result['sign_id']}_{result['source']}.mp4"
                filename = filename.replace(" ", "_")
                save_path = sign_dir / filename
                if download_video(result["video_url"], save_path):
                    download_progress["log"].append(f"  Heruntergeladen: {filename}")
                time.sleep(0.3)

            download_progress["completed_signs"] = i + 1

        download_progress["running"] = False
        download_progress["current_sign"] = ""
        download_progress["log"].append("Download abgeschlossen!")

    thread = threading.Thread(target=run_download, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/download/status")
def api_download_status():
    """Get download progress."""
    return jsonify(download_progress)


@app.route("/api/extract", methods=["POST"])
def api_extract():
    """Start landmark extraction in background."""
    if extraction_progress["running"]:
        return jsonify({"error": "Extraktion läuft bereits"}), 400

    sign_list = load_sign_list()

    def run_extraction():
        extraction_progress["running"] = True
        extraction_progress["total_signs"] = len(sign_list)
        extraction_progress["completed_signs"] = 0

        from src.landmark_extractor import extract_all_landmarks
        extract_all_landmarks(sign_list)

        extraction_progress["running"] = False
        extraction_progress["completed_signs"] = len(sign_list)

    thread = threading.Thread(target=run_extraction, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/extract/status")
def api_extract_status():
    """Get extraction progress."""
    return jsonify(extraction_progress)


@app.route("/api/train", methods=["POST"])
def api_train():
    """Start model training in background."""
    if training_progress["running"]:
        return jsonify({"error": "Training läuft bereits"}), 400

    sign_list = load_sign_list()

    def run_training():
        training_progress["running"] = True
        training_progress["status"] = "Training läuft..."
        training_progress["accuracy"] = None

        from src.trainer import train_model
        success = train_model(sign_list)

        if success:
            training_progress["status"] = "Training abgeschlossen!"
        else:
            training_progress["status"] = "Training fehlgeschlagen - nicht genug Daten"
        training_progress["running"] = False

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/train/status")
def api_train_status():
    """Get training progress."""
    return jsonify(training_progress)


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """Recognize a sign from a webcam frame (sent as base64 image)."""
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Kein Frame erhalten"}), 400

    from src.trainer import load_model
    result = load_model()
    if result is None:
        return jsonify({"error": "Kein Modell geladen"}), 400

    img_data = data["frame"].split(",")[1] if "," in data["frame"] else data["frame"]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Frame konnte nicht dekodiert werden"}), 400

    with HolisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        landmarks = extract_landmarks_from_frame(frame, detector)

    if landmarks is None:
        return jsonify({"prediction": None, "confidence": 0})

    hand_data = landmarks[:126]
    has_hands = bool(np.any(hand_data != 0))

    if not has_hands:
        return jsonify({"prediction": None, "confidence": 0, "has_hands": False})

    return jsonify({
        "landmarks": landmarks.tolist(),
        "has_hands": True,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict a sign from accumulated landmark sequences."""
    data = request.get_json()
    if not data or "sequences" not in data:
        return jsonify({"error": "Keine Sequenzen"}), 400

    from src.trainer import load_model
    result = load_model()
    if result is None:
        return jsonify({"error": "Kein Modell"}), 400

    clf, le = result

    sequences = [np.array(s) for s in data["sequences"]]
    padded = pad_or_truncate_sequence(sequences, LANDMARK_SEQUENCE_LENGTH)
    flat = padded.flatten().reshape(1, -1)

    proba = clf.predict_proba(flat)[0]
    max_idx = np.argmax(proba)
    conf = float(proba[max_idx])

    if conf < 0.3:
        return jsonify({"prediction": None, "confidence": 0})

    prediction = le.inverse_transform([max_idx])[0]
    return jsonify({"prediction": prediction, "confidence": conf})


@app.route("/api/save_recording", methods=["POST"])
def api_save_recording():
    """Save a recorded video from the browser."""
    if "video" not in request.files:
        return jsonify({"error": "Kein Video erhalten"}), 400

    sign_name = request.form.get("sign_name", "")
    if not sign_name:
        return jsonify({"error": "Kein Gebärden-Name"}), 400

    video_file = request.files["video"]
    safe_name = _sanitize_dirname(sign_name)
    sign_dir = RECORDINGS_DIR / safe_name
    sign_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    save_path = sign_dir / f"recording_{timestamp}.webm"
    video_file.save(save_path)

    _convert_webm_to_mp4(save_path)

    return jsonify({
        "status": "saved",
        "recordings": get_recording_count(sign_name),
        "total": get_video_count(sign_name) + get_recording_count(sign_name),
    })


def _convert_webm_to_mp4(webm_path: Path) -> None:
    """Convert a webm recording to mp4 for consistency."""
    import subprocess
    mp4_path = webm_path.with_suffix(".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(webm_path), "-y", str(mp4_path)],
            capture_output=True,
            timeout=30,
        )
        if mp4_path.exists():
            webm_path.unlink()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


@app.route("/api/status")
def api_status():
    """Get status of all signs."""
    sign_list = load_sign_list()
    signs_data = []
    for sign_name in sign_list:
        videos = get_video_count(sign_name)
        recordings = get_recording_count(sign_name)
        total = videos + recordings
        signs_data.append({
            "name": sign_name,
            "videos": videos,
            "recordings": recordings,
            "total": total,
            "needed": max(0, MIN_VIDEOS_PER_SIGN - total),
        })
    return jsonify(signs_data)


def run_webapp(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    """Start the Flask web application."""
    ensure_dirs()
    print(f"\n{'='*50}")
    print("  ÖGS Gebärden-Erkennung")
    print(f"  Web-Interface: http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(host=host, port=port, debug=debug)
