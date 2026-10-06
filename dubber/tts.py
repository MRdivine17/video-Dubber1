"""Step 6 — give every English line a voice.

Default: Coqui XTTS v2 voice cloning. For each speaker we cut a ~20 s reference from
their own isolated speech (the Demucs vocal stem), so the English voice sounds like
the original speaker. Fallback / alternative: edge-tts neural voices, picked by the
speaker's pitch (male / female) and kept distinct per speaker.

Every clip is written to disk as soon as it is made, so an interrupted run resumes
where it stopped.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from .reporting import ProgressLike, log

from .gpu import DEVICE
from .media import decode_to_array, free_gpu, trim_silence

EDGE_VOICES = {
    "male": ["en-US-AndrewMultilingualNeural", "en-US-BrianMultilingualNeural", "en-GB-RyanNeural", "en-US-GuyNeural"],
    "female": ["en-US-AvaMultilingualNeural", "en-US-EmmaMultilingualNeural", "en-GB-SoniaNeural", "en-US-JennyNeural"],
}
CLIP_SR = 24_000


def clip_path(clips_dir: Path, i: int) -> Path:
    return clips_dir / f"{i:05d}.wav"


# ── speaker references ──────────────────────────────────────────────────────────

def build_references(vocals_path: Path, segments: list[dict], labels: list[str], out_dir: Path,
                     target_seconds: float = 20.0) -> dict[str, dict]:
    """Cut a clean reference recording per speaker and estimate their pitch / gender."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = sf.info(str(vocals_path))
    speakers: dict[str, dict] = {}
    for speaker in sorted(set(labels), key=lambda s: int(s[1:])):
        mine = [s for s, lab in zip(segments, labels) if lab == speaker]
        # Longest clean-ish lines first; very long ones are capped to keep the reference varied.
        mine.sort(key=lambda s: -(min(s["end"] - s["start"], 12.0)))
        pieces, total = [], 0.0
        for seg in mine:
            if total >= target_seconds:
                break
            a, b = seg["start"], min(seg["end"], seg["start"] + 12.0)
            if b - a < 1.0 and pieces:
                continue
            audio, _ = sf.read(str(vocals_path), start=int(a * info.samplerate), stop=int(b * info.samplerate),
                               dtype="float32", always_2d=True)
            pieces.append(audio.mean(1))
            pieces.append(np.zeros(int(0.25 * info.samplerate), dtype=np.float32))
            total += b - a
        ref = np.concatenate(pieces) if pieces else np.zeros(info.samplerate, dtype=np.float32)
        peak = np.max(np.abs(ref)) + 1e-9
        ref = ref / peak * 0.9
        path = out_dir / f"{speaker}.wav"
        sf.write(str(path), ref, info.samplerate)
        pitch = estimate_pitch(ref, info.samplerate)
        speakers[speaker] = {
            "reference": str(path),
            "reference_seconds": round(total, 1),
            "pitch_hz": round(pitch, 1),
            "gender": "female" if pitch >= 165 else "male",
            "lines": len(mine),
        }
    return speakers


def estimate_pitch(audio: np.ndarray, sr: int) -> float:
    """Median fundamental frequency of voiced frames (simple autocorrelation)."""
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    f0 = []
    for start in range(0, max(0, len(audio) - frame), hop)[:1500]:
        x = audio[start:start + frame]
        if np.sqrt(np.mean(x ** 2)) < 0.03:
            continue
        x = x - x.mean()
        spec = np.fft.rfft(x, 2 * frame)
        ac = np.fft.irfft(spec * np.conj(spec))[:frame]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.35:
            f0.append(sr / lag)
    return float(np.median(f0)) if f0 else 150.0


# ── synthesis ───────────────────────────────────────────────────────────────────

def clean_for_speech(text: str) -> str:
    text = re.sub(r"[\[\(\{].*?[\]\)\}]", " ", text)        # drop bracketed notes
    text = re.sub(r"[\"“”«»*_#~|<>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def synthesize(lines: list[str], labels: list[str], speakers: dict[str, dict], clips_dir: Path,
               mode: str, progress: ProgressLike) -> str:
    """Create one wav per line in clips_dir. Returns the engine actually used."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(len(lines)) if not clip_path(clips_dir, i).exists()]
    if not todo:
        return mode
    if mode == "clone":
        # No silent fallback: a cloud stock voice is not the speaker's voice and not local,
        # so a cloning failure stops the run with the real error. Finished clips stay cached.
        _synthesize_xtts(lines, labels, speakers, clips_dir, todo, progress)
        return "clone"
    _synthesize_edge(lines, labels, speakers, clips_dir, todo, progress)
    return "edge"


def _synthesize_xtts(lines, labels, speakers, clips_dir: Path, todo: list[int], progress: ProgressLike) -> None:
    import torch
    from TTS.api import TTS

    device = DEVICE
    task = progress.add_task("load XTTS v2", total=None)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    model = tts.synthesizer.tts_model
    sr = model.config.audio.output_sample_rate

    # Library defaults only look at 10 s of reference; using the full ~20-30 s clones the voice more faithfully.
    latents = {}
    for name, spk in speakers.items():
        latents[name] = model.get_conditioning_latents(
            audio_path=[spk["reference"]], max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=4,
        )
    progress.update(task, total=1, completed=1, description=f"XTTS ready ({len(latents)} cloned voice(s))")

    task = progress.add_task("synthesize (cloned voice)", total=len(todo))
    try:
        for i in todo:
            text = clean_for_speech(lines[i]) or "..."
            gpt, emb = latents[labels[i]]
            wav = _xtts_once(model, text, gpt, emb, temperature=0.65)
            # XTTS occasionally rambles on short inputs; retry cooler if the clip is far too long.
            expected = max(1.0, len(text.split()) / 2.5)
            if len(wav) / sr > expected * 2.2 + 1.5:
                retry = _xtts_once(model, text, gpt, emb, temperature=0.3)
                if len(retry) < len(wav):
                    wav = retry
            wav = trim_silence(wav, sr)
            sf.write(str(clip_path(clips_dir, i)), wav, sr)
            progress.advance(task)
    finally:
        del tts, model, latents
        free_gpu()


def _xtts_once(model, text: str, gpt_latent, speaker_embedding, temperature: float) -> np.ndarray:
    out = model.inference(
        text, "en", gpt_latent, speaker_embedding,
        temperature=temperature, length_penalty=1.0, repetition_penalty=5.0,
        top_k=50, top_p=0.85, enable_text_splitting=True,
    )
    return np.asarray(out["wav"], dtype=np.float32)


def _assign_edge_voices(speakers: dict[str, dict]) -> dict[str, str]:
    used = {"male": 0, "female": 0}
    voices = {}
    for name, spk in speakers.items():
        pool = EDGE_VOICES[spk["gender"]]
        voices[name] = pool[used[spk["gender"]] % len(pool)]
        used[spk["gender"]] += 1
    return voices


def _synthesize_edge(lines, labels, speakers, clips_dir: Path, todo: list[int], progress: ProgressLike) -> None:
    import edge_tts

    voices = _assign_edge_voices(speakers)
    log("edge-tts voices: " + ", ".join(f"{k}={v}" for k, v in voices.items()))
    task = progress.add_task("synthesize (edge-tts)", total=len(todo))
    tmp_dir = clips_dir / "_mp3"
    tmp_dir.mkdir(exist_ok=True)

    async def one(i: int, sem: asyncio.Semaphore) -> None:
        async with sem:
            text = clean_for_speech(lines[i]) or "..."
            mp3 = tmp_dir / f"{i:05d}.mp3"
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(text, voices[labels[i]]).save(str(mp3))
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))
            wav = trim_silence(decode_to_array(mp3, CLIP_SR), CLIP_SR)
            sf.write(str(clip_path(clips_dir, i)), wav, CLIP_SR)
            os.remove(mp3)
            progress.advance(task)

    async def main() -> None:
        sem = asyncio.Semaphore(8)
        await asyncio.gather(*(one(i, sem) for i in todo))

    asyncio.run(main())
