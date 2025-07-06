import os
import time
from typing import List, Tuple

import requests
from requests.exceptions import RequestException
from urllib.parse import urlparse
from pathlib import Path

def sanitize_filename(text):
    """Sanitize folder names by removing illegal characters."""
    return "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()

def get_next_script_folder(base_dir='images'):
    Path(base_dir).mkdir(exist_ok=True)
    existing = [f for f in os.listdir(base_dir) if f.startswith('script_') and os.path.isdir(os.path.join(base_dir, f))]
    numbers = [int(f.split('_')[1]) for f in existing if f.split('_')[1].isdigit()]
    next_num = max(numbers, default=0) + 1
    return os.path.join(base_dir, f'script_{next_num}')


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

def download_image(url: str, save_path: str, retries: int = 3) -> bool:
    if not is_valid_url(url):
        print(f"[WARN] Invalid URL skipped: {url}")
        return False

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1]
            if not ext or len(ext) > 5:
                ext = '.jpg'

            with open(save_path + ext, 'wb') as f:
                f.write(response.content)
            return True
        except RequestException as e:
            if attempt == retries:
                print(f"[!] Failed to download {url}: {e}")
                return False
            time.sleep(2 ** attempt)

def download_images_for_script(
    results, images_per_scene: int = 4, base_dir: str = 'images'
) -> Tuple[str, List[str]]:
    """
    results: list of tuples (script_line, [list of image urls])
    """
    script_dir = get_next_script_folder(base_dir)
    os.makedirs(script_dir, exist_ok=True)

    failed: List[str] = []

    for line, urls in results:
        clean_line = sanitize_filename(line.strip())
        if not clean_line:
            continue

        scene_folder = os.path.join(script_dir, clean_line)
        os.makedirs(scene_folder, exist_ok=True)

        for i, url in enumerate(urls[:images_per_scene]):
            save_name = os.path.join(scene_folder, f"image_{i+1:02d}")
            success = download_image(url, save_name)
            if not success:
                print(f"[!] Skipped downloading for line: {line}")
                failed.append(url)

    print(f"✅ All images downloaded to: {script_dir}")
    return script_dir, failed
