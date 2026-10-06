"""Step 7 — place every English clip on the original timeline and mix it over the background.

Timing rules for each line:
* It starts where the original line started (or right after the previous dubbed line,
  if that one ran long).
* If it is longer than the time until the next line, it is sped up (pitch preserved,
  ffmpeg rubberband) up to `max_speed`; anything beyond that spills into the gap and
  the next line shifts slightly — drift is recovered at the next pause.
* If it is much shorter than the original line, it is slowed a touch (min 0.9x) so the
  voice covers the speaker's mouth movement.
* Its loudness follows the original speaker's loudness on that line ("same energy").
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
from .reporting import ProgressLike

from .config import MIX_SR
from .media import change_tempo, rms, soft_limit
from .tts import clip_path

GAP = 0.08        # breathing room left before the next line
MIN_SPEED = 0.9


def build_voice_track(segments: list[dict], clips_dir: Path, vocals_path: Path, total_seconds: float,
                      progress: ProgressLike, max_speed: float) -> tuple[np.ndarray, list[dict]]:
    import torch
    import torchaudio.functional as AF

    sr = MIX_SR
    track = np.zeros(int((total_seconds + 5) * sr), dtype=np.float32)
    vocals_info = sf.info(str(vocals_path))
    original_levels = [_level(vocals_path, vocals_info.samplerate, s["start"], s["end"]) for s in segments]
    typical = float(np.median([lv for lv in original_levels if lv > 0] or [0.05]))

    placements = []
    cursor = 0.0
    task = progress.add_task("align & time-fit lines", total=len(segments))
    for i, seg in enumerate(segments):
        clip, clip_sr = sf.read(str(clip_path(clips_dir, i)), dtype="float32", always_2d=True)
        clip = clip.mean(1)
        if clip_sr != sr:
            clip = AF.resample(torch.from_numpy(clip), clip_sr, sr).numpy()

        start = max(seg["start"], cursor)
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else total_seconds
        window = max(0.3, next_start - start - GAP)
        spoken = seg["end"] - seg["start"]
        natural = len(clip) / sr

        if natural > window:
            factor = min(natural / window, max_speed)
        elif natural < spoken * MIN_SPEED:
            factor = max(natural / spoken, MIN_SPEED)
        else:
            factor = 1.0
        clip = change_tempo(clip, sr, factor)

        # Match the original speaker's loudness on this line, smoothed towards the speaker-wide level.
        target = 0.6 * original_levels[i] + 0.4 * typical if original_levels[i] > 0 else typical
        gain = np.clip(target / (rms(clip) + 1e-9), 0.1, 8.0)
        clip = clip * gain

        a = int(start * sr)
        b = min(len(track), a + len(clip))
        track[a:b] += clip[: b - a]
        end = start + len(clip) / sr
        cursor = end + 0.02
        placements.append({
            "id": i, "start": round(start, 3), "end": round(end, 3),
            "original_start": seg["start"], "original_end": seg["end"],
            "speed": round(factor, 3), "late_by": round(start - seg["start"], 3),
            "overflow": round(max(0.0, end - next_start), 3),
        })
        progress.advance(task)
    return track[: int(total_seconds * sr)], placements


def _level(path: Path, sr: int, start: float, end: float) -> float:
    audio, _ = sf.read(str(path), start=int(start * sr), stop=int(end * sr), dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if len(audio) == 0:
        return 0.0
    # Loudness of the voiced part only (ignore pauses inside the line).
    frame = int(0.02 * sr)
    n = len(audio) // frame
    if n == 0:
        return rms(audio)
    frames = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    voiced = frames[frames > frames.max() * 0.1]
    return float(np.sqrt(np.mean(voiced ** 2))) if len(voiced) else 0.0


def mix_with_background(voice: np.ndarray, background_path: Path, out_path: Path, progress: ProgressLike,
                        background_gain: float = 1.0) -> None:
    """Stream the background stem in blocks, add the dubbed voice, limit peaks, write 16-bit stereo."""
    info = sf.info(str(background_path))
    sr = info.samplerate
    assert sr == MIX_SR
    tmp = out_path.with_suffix(".tmp.wav")
    block = sr * 60
    task = progress.add_task("mix voice + background", total=info.frames)
    with sf.SoundFile(str(background_path)) as bg, sf.SoundFile(str(tmp), "w", sr, 2, "PCM_16") as out:
        pos = 0
        while True:
            chunk = bg.read(block, dtype="float32", always_2d=True)
            if len(chunk) == 0:
                break
            v = voice[pos: pos + len(chunk)]
            if len(v) < len(chunk):
                v = np.pad(v, (0, len(chunk) - len(v)))
            mixed = chunk * background_gain + v[:, None]
            out.write(soft_limit(mixed))
            pos += len(chunk)
            progress.update(task, completed=pos)
    os.replace(tmp, out_path)


def _timestamp(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_subtitles(lines: list[str], placements: list[dict], srt_path: Path, vtt_path: Path) -> None:
    """English subtitles timed to the dubbed speech: .srt for players, .vtt for the browser."""
    cues = list(zip(lines, placements))
    srt = [f"{n}\n{_timestamp(p['start'], ',')} --> {_timestamp(p['end'], ',')}\n{text}\n"
           for n, (text, p) in enumerate(cues, start=1)]
    vtt = [f"{_timestamp(p['start'], '.')} --> {_timestamp(p['end'], '.')}\n{text}\n" for text, p in cues]
    srt_path.write_text("\n".join(srt), encoding="utf-8")
    vtt_path.write_text("WEBVTT\n\n" + "\n".join(vtt), encoding="utf-8")
