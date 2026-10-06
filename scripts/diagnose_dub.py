"""Find where speech content is lost in a finished dub, stage by stage.

    python scripts\\diagnose_dub.py work\\<video id>\\<run> [--tts-sample 40]

1. Recognition coverage: speech energy in the isolated vocals that no transcript line covers.
2. Translation completeness: English words per source word, per line.
3. Synthesis fidelity: Whisper re-transcribes a sample of the English clips and compares
   them with the script (catches the TTS skipping words).
"""
import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber.config import DEFAULT_WHISPER_MODEL, MODELS_DIR, configure_environment  # noqa: E402

configure_environment()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def recognition_coverage(work: Path, segments: list[dict]) -> None:
    audio, sr = sf.read(str(work / "vocals_16k.wav"), dtype="float32")
    hop = int(0.05 * sr)
    n = len(audio) // hop
    energy = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1))
    speech = energy > max(0.01, np.percentile(energy, 60) * 0.5)
    covered = np.zeros(n, dtype=bool)
    for s in segments:
        covered[max(0, int((s["start"] - 0.3) / 0.05)): int((s["end"] + 0.3) / 0.05)] = True
    missed = speech & ~covered
    # Group missed frames into gaps longer than 1.5 s (short blips are breaths / noise).
    gaps, start = [], None
    for i, m in enumerate(np.append(missed, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) * 0.05 >= 1.5:
                gaps.append((start * 0.05, i * 0.05))
            start = None
    speech_s, missed_s = speech.sum() * 0.05, sum(b - a for a, b in gaps)
    print(f"1. RECOGNITION: speech-like audio {speech_s / 60:.1f} min, transcript covers "
          f"{sum(s['end'] - s['start'] for s in segments) / 60:.1f} min")
    print(f"   uncovered speech gaps >= 1.5 s: {len(gaps)} gaps, {missed_s / 60:.1f} min "
          f"({100 * missed_s / max(speech_s, 1):.1f}% of speech)")
    for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:6]:
        print(f"     {a / 60:6.2f}-{b / 60:6.2f} min  ({b - a:.1f} s)")


def translation_completeness(rows: list[dict]) -> None:
    ratios = []
    for r in rows:
        src, en = len(r["source"].split()), len(r["english"].split())
        if src >= 4:
            ratios.append((en / src, r))
    values = np.array([x for x, _ in ratios])
    short = [r for x, r in ratios if x < 0.9]
    print(f"\n2. TRANSLATION: English/source word ratio median {np.median(values):.2f} "
          f"(Tamil -> English is normally ~1.3-2.0)")
    print(f"   lines likely missing content (ratio < 0.9): {len(short)} of {len(ratios)} "
          f"({100 * len(short) / len(ratios):.0f}%)")
    for r in sorted(short, key=lambda r: len(r["english"].split()) / len(r["source"].split()))[:4]:
        print(f"     {r['start']:7.1f}s  {len(r['source'].split())} src words -> {len(r['english'].split())} en: "
              f"{r['english'][:90]}")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def synthesis_fidelity(work: Path, rows: list[dict], sample: int) -> None:
    import torch  # noqa: F401
    from faster_whisper import WhisperModel

    model = WhisperModel(DEFAULT_WHISPER_MODEL, device="cuda", compute_type="int8_float16",
                         download_root=str(MODELS_DIR / "whisper"))
    random.seed(7)
    picked = random.sample(rows, min(sample, len(rows)))
    scores, low = [], []
    for r in picked:
        clip = work / "clips" / f"{r['id']:05d}.wav"
        if not clip.exists():
            continue
        segs, _ = model.transcribe(str(clip), language="en", beam_size=5, vad_filter=False)
        heard = " ".join(s.text for s in segs)
        ratio = difflib.SequenceMatcher(None, words(r["english"]), words(heard)).ratio()
        scores.append(ratio)
        if ratio < 0.75:
            low.append((ratio, r, heard))
    print(f"\n3. SYNTHESIS: re-transcribed {len(scores)} English clips; mean word match {np.mean(scores):.2f}")
    print(f"   clips that lost words (match < 0.75): {len(low)} of {len(scores)}")
    for ratio, r, heard in sorted(low, key=lambda x: x[0])[:4]:
        print(f"     {ratio:.2f}  script ({len(r['english'])} chars): {r['english'][:80]}")
        print(f"           heard: {heard.strip()[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--tts-sample", type=int, default=40)
    args = parser.parse_args()
    transcript = json.loads((args.work / "transcript.json").read_text(encoding="utf-8"))
    rows = json.loads((args.work / "translation.json").read_text(encoding="utf-8"))
    recognition_coverage(args.work, transcript["segments"])
    translation_completeness(rows)
    if args.tts_sample:
        synthesis_fidelity(args.work, rows, args.tts_sample)
