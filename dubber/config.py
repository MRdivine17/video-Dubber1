"""Paths and process-wide settings.

All heavy data (model weights, work files, outputs) lives inside the project
folder so nothing spills onto the system drive.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
WORK_DIR = ROOT / "work"
OUTPUT_DIR = ROOT / "output"
MODELS_READY_MARKER = MODELS_DIR / ".ready"   # written by scripts/download_models.py

# Audio formats used between stages.
MIX_SR = 44_100      # Demucs native rate; also the final mix rate
ASR_SR = 16_000      # Whisper / speaker-embedding input rate

# faster-whisper's built-in name -> Systran/faster-whisper-large-v3 on Hugging Face.
DEFAULT_WHISPER_MODEL = "large-v3"


def configure_environment() -> None:
    """Point every model cache at MODELS_DIR. Must run before ML libraries are imported."""
    for d in (MODELS_DIR, WORK_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # Keys / tokens from <project>/.env (HF_TOKEN, LLM keys). Real environment variables win.
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    defaults = {
        "HF_HOME": MODELS_DIR / "huggingface",
        "TORCH_HOME": MODELS_DIR / "torch",
        "TTS_HOME": MODELS_DIR / "tts",
        "XDG_DATA_HOME": MODELS_DIR / "xdg-data",
        "XDG_CACHE_HOME": MODELS_DIR / "xdg-cache",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Once scripts/download_models.py has fetched everything, never touch the network for models:
    # no update checks, and dubbing keeps working through DNS / connection drops.
    if MODELS_READY_MARKER.exists() and os.environ.get("DUB_ONLINE") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # XTTS v2 weights ship under the Coqui Public Model License (non-commercial).
    os.environ.setdefault("COQUI_TOS_AGREED", "1")


@dataclass
class Options:
    """Everything a single dubbing run can be configured with (see dub.py --help)."""

    url: str
    language: str | None = None          # source language code; None = auto-detect
    voice: str = "clone"                 # "clone" (XTTS v2) or "edge" (edge-tts stock voices)
    speakers: int | None = None          # None = auto-detect number of speakers
    translator: str = "auto"             # "speech" (from audio), "text" (from transcript) or "auto"
    polish: bool = True                  # LLM pass for natural, time-fitted English
    llm_model: str | None = None         # polish model override; None = auto-detect (dubber/llm.py)
    whisper_model: str = DEFAULT_WHISPER_MODEL
    max_minutes: float | None = None     # dub only the first N minutes (quick tests)
    max_speed: float = 1.35              # fastest we will speed up a dubbed line
    background_gain: float = 1.0         # music / effects level in the final mix
    cookies_browser: str | None = None   # pass browser cookies to yt-dlp if YouTube asks
    force: bool = False                  # ignore cached stage outputs
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
