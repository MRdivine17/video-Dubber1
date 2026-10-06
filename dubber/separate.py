"""Step 2 — split the soundtrack into speech and background (music + effects) with Demucs.

Dubbing only the speech and keeping the original background is what preserves the
"energy" of the video. Long audio is processed in 5-minute chunks (with a little context
on each side) so a 2-hour video never has to fit in GPU memory at once. Every finished
chunk is saved, so a stopped run resumes at the first unfinished chunk.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from .gpu import DEVICE
from .media import free_gpu
from .reporting import ProgressLike, log


def separate_vocals(mix_path: Path, vocals_path: Path, background_path: Path, parts_dir: Path,
                    progress: ProgressLike, chunk_seconds: int = 300, context_seconds: int = 3) -> None:
    info = sf.info(str(mix_path))
    sr, total = info.samplerate, info.frames
    chunk, ctx = chunk_seconds * sr, context_seconds * sr
    starts = list(range(0, total, chunk))
    parts_dir.mkdir(parents=True, exist_ok=True)

    finished = {i for i in range(len(starts)) if (parts_dir / f"{i:04d}.done").exists()}
    task = progress.add_task("separate speech / background", total=total / sr)
    if finished:
        log(f"Resuming separation: {len(finished)} of {len(starts)} chunks already done")
        progress.update(task, advance=sum(min(chunk, total - starts[i]) for i in finished) / sr)

    if len(finished) < len(starts):
        _separate_chunks(mix_path, starts, finished, total, sr, chunk, ctx, parts_dir, progress, task)

    # Join the chunks into the two full-length tracks.
    for kind, final in (("vocals", vocals_path), ("background", background_path)):
        tmp = final.with_suffix(".tmp.wav")
        with sf.SoundFile(str(tmp), "w", sr, 2, "PCM_16") as out:
            for i in range(len(starts)):
                out.write(sf.read(str(_part(parts_dir, i, kind)), dtype="int16", always_2d=True)[0])
        os.replace(tmp, final)
    shutil.rmtree(parts_dir, ignore_errors=True)


def _part(parts_dir: Path, i: int, kind: str) -> Path:
    return parts_dir / f"{i:04d}.{kind}.wav"


def _separate_chunks(mix_path: Path, starts: list[int], finished: set[int], total: int, sr: int,
                     chunk: int, ctx: int, parts_dir: Path, progress: ProgressLike, task: int) -> None:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.to(DEVICE).eval()
    if sr != model.samplerate:
        raise ValueError(f"expected {model.samplerate} Hz input, got {sr}")
    vocal_idx = model.sources.index("vocals")

    for i, start in enumerate(starts):
        if i in finished:
            continue
        a, b = max(0, start - ctx), min(total, start + chunk + ctx)
        mix, _ = sf.read(str(mix_path), start=a, stop=b, dtype="float32", always_2d=True)
        wav = torch.from_numpy(mix.T.copy()).to(DEVICE)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(model, ((wav - mean) / std)[None], device=DEVICE,
                                  split=True, overlap=0.25, progress=False)[0]
        vocals = sources[vocal_idx] * std
        background = wav - vocals            # everything that is not speech
        keep = slice(start - a, start - a + min(chunk, total - start))
        sf.write(str(_part(parts_dir, i, "vocals")), np.clip(vocals[:, keep].T.cpu().numpy(), -1, 1), sr,
                 subtype="PCM_16")
        sf.write(str(_part(parts_dir, i, "background")), np.clip(background[:, keep].T.cpu().numpy(), -1, 1),
                 sr, subtype="PCM_16")
        (parts_dir / f"{i:04d}.done").touch()   # written last: marks the chunk complete
        progress.update(task, advance=(keep.stop - keep.start) / sr)
        del sources, vocals, background, wav

    del model
    free_gpu()
