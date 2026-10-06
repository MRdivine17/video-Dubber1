"""Step 3 — speech-to-text with faster-whisper, then regroup words into sentences.

Whisper's own segments often cut sentences in half. Translation and dubbing work far
better on whole sentences, so we ask for word timestamps and rebuild sentence-sized
units with their exact start / end times.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .config import ASR_SR, MODELS_DIR
from .gpu import DEVICE
from .media import free_gpu, read_json, write_json
from .reporting import ProgressLike, fmt_seconds, log

SENTENCE_END = (".", "?", "!", "।", "॥", "。", "？", "！", "…", "؟")
SOFT_BREAK = (",", "،", "、", "，", ";", ":")


# Each batch item keeps beam_size decoding streams (plus encoder cross-attention) in VRAM.
# If that exceeds the card, Windows silently spills into system RAM and decoding slows ~10x,
# so the batch is sized from the free memory, and 2 beams keep accuracy close to 5 at half the cost.
BEAM_SIZE = 2


def _batch_size_for_free_vram() -> int:
    import torch

    free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3   # measured after the model is loaded
    if free_gb >= 12:
        return 16
    if free_gb >= 4.5:
        return 8      # ~3.7 GB at beam 2 for large-v3
    if free_gb >= 3:
        return 4
    return 2


@dataclass
class Word:
    start: float
    end: float
    text: str


def transcribe(audio_16k: Path, language: str | None, model_name: str, progress: ProgressLike,
               total_seconds: float, checkpoint: Path) -> tuple[list[dict], str]:
    """Transcribe `audio_16k`. Progress is saved to `checkpoint` every few seconds; if it exists,
    only the audio after the last saved point is transcribed."""
    import torch  # noqa: F401  (loads the CUDA / cuDNN DLLs ctranslate2 needs on Windows)
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    words: list[Word] = []
    offset = 0.0
    source = audio_16k
    if checkpoint.exists():
        state = read_json(checkpoint)
        words = [Word(*w) for w in state["words"]]
        offset = state["done_until"]
        language = language or state["language"]
        log(f"Resuming transcription at {fmt_seconds(offset)} of {fmt_seconds(total_seconds)}")
        audio, sr = sf.read(str(audio_16k), start=int(offset * ASR_SR), dtype="float32")
        source = audio_16k.with_name("vocals_16k.resume.wav")
        sf.write(str(source), audio, sr)

    loading = progress.add_task(f"load Whisper {model_name} onto the GPU", total=None)
    # int8 weights + fp16 activations: half the VRAM of pure fp16, same accuracy, faster on tensor cores.
    model = WhisperModel(model_name, device=DEVICE,
                         compute_type="int8_float16",
                         download_root=str(MODELS_DIR / "whisper"))
    batch_size = _batch_size_for_free_vram()
    progress.update(loading, total=1, completed=1,
                    description=f"Whisper {model_name} ready on the GPU (batch {batch_size})")
    pipeline = BatchedInferencePipeline(model)
    segments, info = pipeline.transcribe(
        str(source),
        language=language,
        batch_size=batch_size,
        beam_size=BEAM_SIZE,
        word_timestamps=True,
        vad_filter=True,
    )
    language = language or info.language
    task = progress.add_task(f"transcribe ({language})", total=total_seconds)
    progress.update(task, completed=offset)

    def save(done_until: float) -> None:
        write_json(checkpoint, {"language": language, "done_until": done_until,
                                "words": [[w.start, w.end, w.text] for w in words]})

    last_save = time.monotonic()
    for seg in segments:  # generator: decoding happens while we iterate (segments arrive in time order)
        if seg.words:
            words.extend(Word(offset + w.start, offset + w.end, w.word) for w in seg.words)
        else:
            words.append(Word(offset + seg.start, offset + seg.end, seg.text))
        done_until = offset + seg.end
        progress.update(task, completed=min(done_until, total_seconds))
        if time.monotonic() - last_save > 15:
            save(done_until)
            last_save = time.monotonic()
    progress.update(task, completed=total_seconds)

    # The exhausted generator still references the pipeline (and so the model): drop it too,
    # and unload the weights explicitly so the next step gets the whole GPU.
    del segments, pipeline
    model.model.unload_model()
    del model
    free_gpu()
    if source != audio_16k:
        source.unlink(missing_ok=True)
    return group_into_sentences(words), language


def group_into_sentences(words: list[Word], max_seconds: float = 12.0, pause_split: float = 0.6) -> list[dict]:
    """Merge words into sentence-sized lines, split on punctuation, pauses and length."""
    sentences: list[dict] = []
    current: list[Word] = []

    def flush() -> None:
        text = "".join(w.text for w in current).strip()
        if text.strip(".,!?…-— "):
            sentences.append({"start": round(current[0].start, 3), "end": round(current[-1].end, 3), "text": text})
        current.clear()

    for i, word in enumerate(words):
        current.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        token = word.text.strip()
        length = word.end - current[0].start
        pause = (nxt.start - word.end) if nxt else float("inf")
        if (
            nxt is None
            or pause > pause_split
            or (token.endswith(SENTENCE_END) and length >= 1.5)
            or (token.endswith(SOFT_BREAK) and length >= max_seconds * 0.6)
            or length >= max_seconds
        ):
            flush()
    return sentences
