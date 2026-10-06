"""Speech translation: Whisper translates each line's audio straight into English.

Why this exists: for Indian languages, recognise-then-translate loses content. The text
step only sees the recognised Tamil / Hindi text, and conversational speech is full of
English words ("code-mixing") that the recogniser writes in the native script and the text
model then mistranslates or drops. Translating from the audio keeps those English words as
spoken and translates every sentence the speaker actually said.

Each transcript line is decoded as its own batch item, on its exact time span, so the
English stays aligned with the original timing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .config import ASR_SR, MODELS_DIR
from .gpu import DEVICE
from .media import free_gpu, read_json, write_json
from .reporting import ProgressLike, log

PAD = 0.15            # seconds of context on each side of a line
GROUP = 64            # lines per decode call; progress is saved after each group
MAX_CLIP = 29.5       # Whisper's window is 30 s
BEAM_SIZE = 3


def translate_speech(audio_16k: Path, segments: list[dict], language: str, model_name: str,
                     progress: ProgressLike, checkpoint: Path) -> list[str]:
    """English for every segment, translated from that segment's audio."""
    import torch  # noqa: F401  (loads the CUDA DLLs ctranslate2 needs on Windows)
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    done: dict[str, str] = read_json(checkpoint) if checkpoint.exists() else {}
    todo = [i for i in range(len(segments)) if str(i) not in done]
    task = progress.add_task("translate speech to English (Whisper)", total=len(segments))
    progress.advance(task, len(segments) - len(todo))
    if done:
        log(f"Resuming speech translation: {len(done)} of {len(segments)} lines already done")

    if todo:
        audio, sr = sf.read(str(audio_16k), dtype="float32")
        assert sr == ASR_SR
        duration = len(audio) / sr
        model = WhisperModel(model_name, device=DEVICE, compute_type="int8_float16",
                             download_root=str(MODELS_DIR / "whisper"))
        pipeline = BatchedInferencePipeline(model)
        try:
            for k in range(0, len(todo), GROUP):
                group = todo[k:k + GROUP]
                spans = [_span(segments[i], duration) for i in group]
                result, _ = pipeline.transcribe(
                    audio, task="translate", language=language,
                    clip_timestamps=[{"start": a, "end": b} for a, b in spans],
                    batch_size=8, beam_size=BEAM_SIZE, without_timestamps=True,
                    condition_on_previous_text=False,
                )
                pieces: dict[int, list[str]] = {i: [] for i in group}
                for seg in result:
                    pieces[_owner(seg.start, group, spans)].append(seg.text.strip())
                for i in group:
                    done[str(i)] = " ".join(p for p in pieces[i] if p).strip()
                write_json(checkpoint, done)
                progress.advance(task, len(group))
        finally:
            del pipeline
            model.model.unload_model()
            del model
            free_gpu()

    return [done.get(str(i), "") for i in range(len(segments))]


def _span(seg: dict, duration: float) -> tuple[float, float]:
    start = max(0.0, seg["start"] - PAD)
    end = min(duration, seg["end"] + PAD, start + MAX_CLIP)
    return round(start, 3), round(end, 3)


def _owner(t: float, group: list[int], spans: list[tuple[float, float]]) -> int:
    """The line a decoded segment belongs to.

    With without_timestamps=True every span yields one segment starting exactly at the span
    start. Padded spans of neighbouring lines overlap slightly, so "which span contains t"
    is ambiguous; the span whose start is nearest t is not.
    """
    return min(zip(group, spans), key=lambda x: abs(x[1][0] - t))[0]


def looks_unreliable(text: str, seconds: float) -> bool:
    """Empty output, a decoder loop, or far too few words for the speech length."""
    words = text.split()
    if not words:
        return True
    if seconds >= 3 and len(words) / seconds < 0.4:
        return True
    distinct = len(set(w.lower().strip(".,!?") for w in words))
    return len(words) >= 8 and distinct / len(words) < 0.35
