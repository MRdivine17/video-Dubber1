"""Thin wrappers around ffmpeg / ffprobe and small audio helpers."""
from __future__ import annotations

import functools
import json
import os
import subprocess
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


def run_ffmpeg(args: list[str], input_bytes: bytes | None = None) -> bytes:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    if input_bytes is not None:
        cmd.remove("-nostdin")
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[-2000:]
        raise FFmpegError(f"ffmpeg failed ({' '.join(cmd[:12])} ...):\n{err}")
    return proc.stdout


def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


def video_codec(path: Path) -> str | None:
    for stream in probe(path)["streams"]:
        if stream.get("codec_type") == "video":
            return stream.get("codec_name")
    return None


def extract_audio(src: Path, dst: Path, sample_rate: int, channels: int) -> None:
    tmp = dst.with_suffix(".tmp.wav")
    run_ffmpeg(["-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", str(tmp)])
    os.replace(tmp, dst)


def trim_video(src: Path, dst: Path, seconds: float) -> None:
    """Cut the first `seconds` without re-encoding (used for quick test runs)."""
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    run_ffmpeg(["-i", str(src), "-t", f"{seconds:.3f}", "-map", "0:v:0", "-map", "0:a:0",
                "-c", "copy", str(tmp)])
    os.replace(tmp, dst)


def mux(video: Path, audio: Path, subtitles: Path | None, dst: Path) -> None:
    """Swap in the dubbed audio. The video stream is copied bit-for-bit (no re-encode)."""
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    args = ["-i", str(video), "-i", str(audio)]
    if subtitles:
        args += ["-i", str(subtitles)]
    args += ["-map", "0:v:0", "-map", "1:a:0"]
    if subtitles:
        args += ["-map", "2:s:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-metadata:s:a:0", "language=eng",
             "-movflags", "+faststart", str(tmp)]
    run_ffmpeg(args)
    os.replace(tmp, dst)


@functools.cache
def has_filter(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def change_tempo(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Speed speech up (factor > 1) or slow it down without changing its pitch."""
    if abs(factor - 1.0) < 0.01 or len(audio) == 0:
        return audio
    if has_filter("rubberband"):
        filt = f"rubberband=tempo={factor:.4f}:window=short"
    else:
        filt = f"atempo={factor:.4f}"
    raw = run_ffmpeg(
        ["-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
         "-af", filt, "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "pipe:1"],
        input_bytes=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
    )
    return np.frombuffer(raw, dtype=np.float32).copy()


def decode_to_array(src: Path, sample_rate: int) -> np.ndarray:
    """Decode any audio file (e.g. edge-tts mp3) to mono float32."""
    raw = run_ffmpeg(["-i", str(src), "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"])
    return np.frombuffer(raw, dtype=np.float32).copy()


def trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0,
                 pad: float = 0.04) -> np.ndarray:
    """Remove leading / trailing silence so clips start exactly where they are placed."""
    if len(audio) == 0:
        return audio
    frame = max(1, int(0.01 * sample_rate))
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms / (np.max(rms) + 1e-12)) > threshold_db)[0]
    if len(loud) == 0:
        return audio
    p = int(pad * sample_rate)
    start = max(0, loud[0] * frame - p)
    end = min(len(audio), (loud[-1] + 1) * frame + p)
    return audio[start:end]


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12)) if len(audio) else 0.0


def soft_limit(x: np.ndarray, knee: float = 0.9) -> np.ndarray:
    """Gentle limiter: untouched below `knee`, smoothly saturates towards 1.0 above it."""
    mag = np.abs(x)
    over = mag > knee
    if np.any(over):
        x = x.copy()
        x[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((mag[over] - knee) / (1 - knee)))
    return x


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
