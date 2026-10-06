"""Automated video dubbing: YouTube URL in, English-dubbed video out.

    python dub.py "https://www.youtube.com/watch?v=..."
    python dub.py URL --max-minutes 2        # quick test on the first 2 minutes
    python dub.py                            # prompts for the URL

The finished video, English subtitles and a timing report are written to ./output.
Re-running the same command resumes from the last finished step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dubber.config import DEFAULT_WHISPER_MODEL, OUTPUT_DIR, Options, configure_environment


def parse_args(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    parser.add_argument("url", nargs="?", help="YouTube video URL (prompted for if omitted)")
    parser.add_argument("--lang", help="source language code, e.g. hi, ta, de, fr (default: auto-detect)")
    parser.add_argument("--voice", choices=["clone", "edge"], default="clone",
                        help="clone = XTTS v2 clone of each speaker (default); edge = edge-tts neural voices")
    parser.add_argument("--speakers", type=int, help="number of speakers (default: auto-detect)")
    parser.add_argument("--translator", choices=["auto", "speech", "text"], default="auto",
                        help="speech = Whisper translates the audio; text = IndicTrans2/NLLB translate the "
                             "transcript; auto = speech for Indian languages, text otherwise (default)")
    parser.add_argument("--no-polish", action="store_true", help="skip the LLM translation polish pass")
    parser.add_argument("--llm-model", help="override the auto-detected LLM model for the polish pass")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="faster-whisper model name")
    parser.add_argument("--max-minutes", type=float, help="only dub the first N minutes (for testing)")
    parser.add_argument("--max-speed", type=float, default=1.35, help="max speed-up applied to a dubbed line")
    parser.add_argument("--background-gain", type=float, default=1.0, help="music / effects level (1.0 = original)")
    parser.add_argument("--cookies-from-browser", help="e.g. chrome, firefox — if YouTube asks to sign in")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="redo every step instead of reusing cached results")
    args = parser.parse_args(argv)

    url = args.url or input("YouTube URL: ").strip()
    if not url:
        parser.error("a YouTube URL is required")
    return Options(
        url=url,
        language=args.lang,
        voice=args.voice,
        speakers=args.speakers,
        translator=args.translator,
        polish=not args.no_polish,
        llm_model=args.llm_model,
        whisper_model=args.whisper_model,
        max_minutes=args.max_minutes,
        max_speed=args.max_speed,
        background_gain=args.background_gain,
        cookies_browser=args.cookies_from_browser,
        force=args.force,
        output_dir=args.output_dir,
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_environment()          # must run before any ML library is imported
    options = parse_args()

    from dubber.gpu import GPUUnavailable
    from dubber.pipeline import run
    from dubber.reporting import log

    try:
        run(options)
    except KeyboardInterrupt:
        log("Interrupted. Run the same command again to resume from the last finished step.", "error")
        return 130
    except GPUUnavailable as exc:
        log(str(exc), "error")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
