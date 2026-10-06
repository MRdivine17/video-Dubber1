"""Step 1 — fetch the YouTube video with yt-dlp."""
from __future__ import annotations

import re
import time
from pathlib import Path

from .media import read_json, write_json
from .reporting import ProgressLike

# Prefer an mp4-compatible video stream (max 1080p) + m4a audio; fall back to anything.
FORMAT = (
    "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/"
    "bv*[height<=1080]+ba/"
    "b[height<=1080]/b"
)
YOUTUBE_URL = re.compile(
    r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com/(watch\?.*v=|shorts/|live/|embed/)|youtu\.be/)[\w-]{6,}",
    re.IGNORECASE,
)


VIDEO_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|live/|embed/)([\w-]{11})")


def is_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_URL.match(url.strip()))


def _base_opts(cookies_browser: str | None) -> dict:
    opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 10,
        "fragment_retries": 10,
        # YouTube needs a JavaScript runtime to unlock all formats; Node is used if Deno is absent.
        "js_runtimes": {"node": {}, "deno": {}},
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    return opts


def _with_retries(fn, attempts: int = 5, on_retry=None):
    import yt_dlp

    for attempt in range(1, attempts + 1):   # survive transient network / DNS drops
        try:
            return fn()
        except yt_dlp.utils.DownloadError:
            if attempt == attempts:
                raise
            if on_retry:
                on_retry(attempt)
            time.sleep(4 * attempt)


def _summary(info: dict, url: str) -> dict:
    return {
        "id": info["id"],
        "title": info.get("title") or info["id"],
        "duration": info.get("duration"),
        "language": info.get("language"),
        "uploader": info.get("uploader") or info.get("channel"),
        "thumbnail": info.get("thumbnail"),
        "url": info.get("webpage_url") or url,
    }


def fetch_info(url: str, cookies_browser: str | None = None) -> dict:
    """Video metadata without downloading (used by the web UI preview)."""
    import yt_dlp

    def run() -> dict:
        with yt_dlp.YoutubeDL(_base_opts(cookies_browser) | {"skip_download": True}) as ydl:
            return ydl.extract_info(url, download=False)

    return _summary(_with_retries(run, attempts=3), url)


def video_id(url: str) -> str | None:
    match = VIDEO_ID.search(url)
    return match.group(1) if match else None


def download(url: str, work_root: Path, progress: ProgressLike,
             cookies_browser: str | None = None) -> tuple[dict, Path]:
    """Download `url` to <work_root>/<video id>/source.mp4. Returns (metadata, path).

    A video already downloaded (source.mp4 + meta.json) is reused without touching the
    network, so resuming never fails on a connection drop.
    """
    import yt_dlp

    task = progress.add_task("download", total=None)
    vid = video_id(url)
    if vid:
        cached, meta_file = work_root / vid / "source.mp4", work_root / vid / "meta.json"
        if cached.exists() and meta_file.exists():
            progress.update(task, total=1, completed=1, description="video already downloaded")
            return read_json(meta_file) | {"downloaded": False}, cached
    downloaded = False

    def hook(d: dict) -> None:
        nonlocal downloaded
        if d["status"] == "downloading":
            downloaded = True
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            kind = "video" if d.get("info_dict", {}).get("vcodec", "none") != "none" else "audio"
            progress.update(task, total=total, completed=d.get("downloaded_bytes", 0),
                            description=f"download {kind} stream")

    opts = _base_opts(cookies_browser) | {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(work_root / "%(id)s" / "source.%(ext)s"),
        "progress_hooks": [hook],
        "concurrent_fragment_downloads": 4,
    }

    def run() -> dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.sanitize_info(ydl.extract_info(url, download=True))

    info = _with_retries(run, on_retry=lambda n: progress.update(task, description=f"download failed, retry {n}/4"))
    path = work_root / info["id"] / "source.mp4"
    if not path.exists():
        raise FileNotFoundError(f"yt-dlp finished but {path} is missing")
    progress.update(task, total=1, completed=1, description="download complete")
    meta = _summary(info, url)
    write_json(path.with_name("meta.json"), meta)
    return meta | {"downloaded": downloaded}, path
