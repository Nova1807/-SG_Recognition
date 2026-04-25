"""Main entry point for the ÖGS sign recognition program."""

import sys

from src.config import ensure_dirs, load_sign_list, MIN_VIDEOS_PER_SIGN
from src.scraper import download_all_signs, get_video_count, get_recording_count


def print_usage() -> None:
    """Print usage information."""
    print(
        """
╔══════════════════════════════════════════════════╗
║     ÖGS Gebärden-Erkennung                      ║
║     Österreichische Gebärdensprache              ║
╚══════════════════════════════════════════════════╝

Verwendung: python -m src.main <befehl>

Befehle:
  download   Videos von gebaerden-archiv.at herunterladen
  extract    Landmarks aus Videos extrahieren
  train      Erkennungsmodell trainieren
  recognize  Live-Erkennung über Webcam starten
  record     Aufnahme-Modus für fehlende Gebärden
  status     Status aller Gebärden anzeigen
  pipeline   Komplette Pipeline (download → extract → train)
"""
    )


def cmd_download(sign_list: list[str]) -> None:
    """Download videos for all signs."""
    print("\n📥 Starte Video-Download...")
    stats = download_all_signs(sign_list)

    print("\n" + "=" * 50)
    print("DOWNLOAD-ZUSAMMENFASSUNG")
    print("=" * 50)
    for sign, count in stats.items():
        status = "OK" if count >= MIN_VIDEOS_PER_SIGN else f"WENIG ({count}/{MIN_VIDEOS_PER_SIGN})"
        print(f"  {sign}: {count} Videos [{status}]")


def cmd_extract(sign_list: list[str]) -> None:
    """Extract landmarks from all videos."""
    from src.landmark_extractor import extract_all_landmarks

    print("\n🔍 Starte Landmark-Extraktion...")
    extract_all_landmarks(sign_list)
    print("\nLandmark-Extraktion abgeschlossen!")


def cmd_train(sign_list: list[str]) -> None:
    """Train the recognition model."""
    from src.trainer import train_model

    print("\n🧠 Starte Modell-Training...")
    success = train_model(sign_list)
    if success:
        print("\nModell-Training erfolgreich abgeschlossen!")
    else:
        print("\nModell-Training fehlgeschlagen.")
        sys.exit(1)


def cmd_recognize() -> None:
    """Start live recognition."""
    from src.recognizer import run_recognition

    run_recognition()


def cmd_record(sign_list: list[str]) -> None:
    """Start recording mode."""
    from src.recorder import run_recording_mode

    run_recording_mode(sign_list)


def cmd_status(sign_list: list[str]) -> None:
    """Show status of all signs."""
    from src.config import LANDMARKS_DIR
    from src.scraper import _sanitize_dirname

    print("\n" + "=" * 60)
    print("STATUS ALLER GEBÄRDEN")
    print("=" * 60)
    print(f"{'Gebärde':<20} {'Videos':>8} {'Aufn.':>8} {'Landmarks':>10} {'Status':>10}")
    print("-" * 60)

    for sign_name in sign_list:
        videos = get_video_count(sign_name)
        recordings = get_recording_count(sign_name)
        total = videos + recordings

        safe_name = _sanitize_dirname(sign_name)
        landmark_dir = LANDMARKS_DIR / safe_name
        landmarks = len(list(landmark_dir.glob("*.npy"))) if landmark_dir.exists() else 0

        if total >= MIN_VIDEOS_PER_SIGN:
            status = "OK"
        elif total > 0:
            status = "WENIG"
        else:
            status = "FEHLT"

        print(f"  {sign_name:<18} {videos:>8} {recordings:>8} {landmarks:>10} {status:>10}")

    print("-" * 60)
    print(f"  Minimum pro Gebärde: {MIN_VIDEOS_PER_SIGN} Videos/Aufnahmen")


def cmd_pipeline(sign_list: list[str]) -> None:
    """Run the complete pipeline: download → extract → train."""
    cmd_download(sign_list)
    cmd_extract(sign_list)
    cmd_train(sign_list)
    print("\n" + "=" * 50)
    print("PIPELINE ABGESCHLOSSEN!")
    print("=" * 50)
    print("Starte jetzt die Live-Erkennung mit: python -m src.main recognize")


def main() -> None:
    """Main entry point."""
    ensure_dirs()

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command in ("recognize",):
        if command == "recognize":
            cmd_recognize()
        return

    sign_list = load_sign_list()
    if not sign_list:
        print("[FEHLER] Keine Gebärden in gebaerden.txt gefunden!")
        sys.exit(1)

    print(f"Geladene Gebärden: {len(sign_list)}")
    for sign in sign_list:
        print(f"  - {sign}")

    commands = {
        "download": lambda: cmd_download(sign_list),
        "extract": lambda: cmd_extract(sign_list),
        "train": lambda: cmd_train(sign_list),
        "record": lambda: cmd_record(sign_list),
        "status": lambda: cmd_status(sign_list),
        "pipeline": lambda: cmd_pipeline(sign_list),
    }

    if command in commands:
        commands[command]()
    else:
        print(f"[FEHLER] Unbekannter Befehl: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
