"""Main entry point for the ÖGS sign recognition program."""

import sys

from src.config import ensure_dirs


def main() -> None:
    ensure_dirs()

    # Standardmäßig wird der Desktop-Modus gestartet
    mode = sys.argv[1] if len(sys.argv) > 1 else "desktop"

    if mode == "live":
        from src.recognizer import run_recognition
        run_recognition()
        return

    if mode == "desktop":
        from src.desktop_app import run_desktop_app
        run_desktop_app()
        return

    print("Verwendung:")
    print("  python -m src.main desktop  -> Startet Desktop-App (Standard)")
    print("  python -m src.main live     -> Startet Live-Erkennung (OpenCV)")


if __name__ == "__main__":
    main()