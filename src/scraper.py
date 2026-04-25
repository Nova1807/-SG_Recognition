"""Scraper for downloading sign language videos from gebaerden-archiv.at."""

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import (
    MEDIA_BASE_URL,
    SEARCH_URL,
    VIDEOS_DIR,
    ensure_dirs,
)


def search_sign_videos(sign_name: str) -> list[dict[str, str]]:
    """Search gebaerden-archiv.at for videos of a specific sign.

    Returns a list of dicts with keys: 'title', 'video_url', 'source', 'sign_id'
    """
    params = {"q": sign_name, "tag": ""}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(SEARCH_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [FEHLER] Suche fehlgeschlagen für '{sign_name}': {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    sign_divs = soup.find_all("div", class_="sign")
    for div in sign_divs:
        title_elem = div.find("h2", class_="meaning")
        if not title_elem:
            continue
        title = title_elem.get_text(strip=True)

        if title.lower() != sign_name.lower():
            continue

        video_elem = div.find("video", class_="signvideo")
        if not video_elem:
            continue

        video_src = video_elem.get("src", "")
        if not video_src:
            source_elem = video_elem.find("source")
            if source_elem:
                video_src = source_elem.get("src", "")

        if not video_src:
            continue

        if video_src.startswith("/"):
            video_url = MEDIA_BASE_URL + video_src
        else:
            video_url = video_src

        source_elem = div.find("div", class_="metadata")
        source_name = "unbekannt"
        if source_elem:
            source_link = source_elem.find("a")
            if source_link:
                source_name = source_link.get_text(strip=True)

        sign_id_match = re.search(r"data-id=[\"']?(\d+)", str(div))
        sign_id = sign_id_match.group(1) if sign_id_match else "unknown"

        results.append({
            "title": title,
            "video_url": video_url,
            "source": source_name,
            "sign_id": sign_id,
        })

    return results


def download_video(video_url: str, save_path: Path) -> bool:
    """Download a single video file."""
    if save_path.exists():
        return True

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(video_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except requests.RequestException as e:
        print(f"  [FEHLER] Download fehlgeschlagen: {video_url} - {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_all_signs(sign_list: list[str]) -> dict[str, int]:
    """Download videos for all signs in the list.

    Returns a dict mapping sign names to the number of downloaded videos.
    """
    ensure_dirs()
    stats: dict[str, int] = {}

    for sign_name in sign_list:
        print(f"\n{'='*50}")
        print(f"Suche Videos für: {sign_name}")
        print(f"{'='*50}")

        sign_dir = VIDEOS_DIR / _sanitize_dirname(sign_name)
        sign_dir.mkdir(parents=True, exist_ok=True)

        existing = list(sign_dir.glob("*.mp4"))
        print(f"  Bereits vorhanden: {len(existing)} Videos")

        results = search_sign_videos(sign_name)
        print(f"  Gefunden: {len(results)} Videos auf gebaerden-archiv.at")

        downloaded = len(existing)
        for result in tqdm(results, desc=f"  Lade '{sign_name}'", unit="video"):
            filename = f"{result['sign_id']}_{result['source']}.mp4"
            filename = _sanitize_filename(filename)
            save_path = sign_dir / filename

            if download_video(result["video_url"], save_path):
                downloaded += 1

            time.sleep(0.5)

        stats[sign_name] = downloaded
        total = len(list(sign_dir.glob("*.mp4")))
        print(f"  Gesamt: {total} Videos für '{sign_name}'")

    return stats


def get_video_count(sign_name: str) -> int:
    """Get the number of downloaded videos for a sign."""
    sign_dir = VIDEOS_DIR / _sanitize_dirname(sign_name)
    if not sign_dir.exists():
        return 0
    return len(list(sign_dir.glob("*.mp4")))


def get_recording_count(sign_name: str) -> int:
    """Get the number of recordings for a sign."""
    from src.config import RECORDINGS_DIR

    sign_dir = RECORDINGS_DIR / _sanitize_dirname(sign_name)
    if not sign_dir.exists():
        return 0
    return len(list(sign_dir.glob("*.mp4")))


def _sanitize_dirname(name: str) -> str:
    """Sanitize a sign name for use as a directory name."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename."""
    return re.sub(r'[<>:"/\\|?*\s]', "_", name).strip()
