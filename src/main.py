"""Main entry point for the ÖGS sign recognition program."""

from src.config import ensure_dirs


def main() -> None:
    """Main entry point - starts the desktop app."""
    ensure_dirs()

    from src.desktop_app import run_desktop_app
    run_desktop_app()


if __name__ == "__main__":
    main()
