"""Step 4 — work out who is speaking each line (speaker diarization).

* If HF_TOKEN is set and pyannote.audio is installed, use pyannote/speaker-diarization-3.1.
* Otherwise: embed every line with SpeechBrain's ECAPA speaker model and cluster the
  embeddings (no account or token needed).

Each speaker later gets their own cloned voice.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
from .reporting import ProgressLike

from .config import ASR_SR, MODELS_DIR
from .gpu import DEVICE
from .reporting import log
from .media import free_gpu

MIN_EMBED_SECONDS = 1.0
CLUSTER_THRESHOLD = 0.55   # cosine distance; lower = more speakers
MIN_SPEAKER_SHARE = 0.05   # clusters with < 5% of speech are merged into the nearest speaker


def diarize(audio_16k: Path, segments: list[dict], num_speakers: int | None, progress: ProgressLike,
            max_speakers: int = 6) -> list[str]:
    if num_speakers == 1 or len(segments) < 2:
        return ["S1"] * len(segments)
    if os.environ.get("HF_TOKEN"):
        try:
            return _pyannote(audio_16k, segments, num_speakers, max_speakers, progress)
        except Exception as exc:  # gated model not accepted, package missing, ...
            log(f"pyannote unavailable ({exc}); using ECAPA clustering", "warn")
    return _ecapa_clustering(audio_16k, segments, num_speakers, max_speakers, progress)


def _pyannote(audio_16k: Path, segments: list[dict], num_speakers: int | None, max_speakers: int,
              progress: ProgressLike) -> list[str]:
    import torch
    from pyannote.audio import Pipeline

    task = progress.add_task("diarize (pyannote)", total=None)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])
    pipeline.to(torch.device(DEVICE))
    kwargs = {"num_speakers": num_speakers} if num_speakers else {"max_speakers": max_speakers}
    annotation = pipeline(str(audio_16k), **kwargs)
    turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]
    labels = []
    for seg in segments:
        overlap: dict[str, float] = {}
        for start, end, spk in turns:
            o = min(seg["end"], end) - max(seg["start"], start)
            if o > 0:
                overlap[spk] = overlap.get(spk, 0.0) + o
        labels.append(max(overlap, key=overlap.get) if overlap else None)
    progress.update(task, total=1, completed=1)
    del pipeline
    free_gpu()
    return _finalise(segments, labels)


def _ecapa_clustering(audio_16k: Path, segments: list[dict], num_speakers: int | None, max_speakers: int,
                      progress: ProgressLike) -> list[str]:
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    device = DEVICE
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODELS_DIR / "speechbrain-ecapa"),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,   # Windows: avoid symlinks
    )
    audio, sr = sf.read(str(audio_16k), dtype="float32")
    assert sr == ASR_SR

    task = progress.add_task("speaker embeddings", total=len(segments))
    embeddings, embedded_idx = [], []
    for i, seg in enumerate(segments):
        if seg["end"] - seg["start"] >= MIN_EMBED_SECONDS:
            clip = audio[int(seg["start"] * sr): int(min(seg["end"], seg["start"] + 10) * sr)]
            with torch.no_grad():
                emb = encoder.encode_batch(torch.from_numpy(clip)[None].to(device)).squeeze().cpu().numpy()
            embeddings.append(emb / (np.linalg.norm(emb) + 1e-9))
            embedded_idx.append(i)
        progress.advance(task)
    del encoder
    free_gpu()

    if len(embeddings) < 2:
        return ["S1"] * len(segments)
    x = np.stack(embeddings)
    if num_speakers:
        model = AgglomerativeClustering(n_clusters=min(num_speakers, len(x)), metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTER_THRESHOLD,
                                        metric="cosine", linkage="average")
    cluster = model.fit_predict(x)

    if not num_speakers:
        cluster = _merge_small_clusters(x, cluster, [segments[i] for i in embedded_idx], max_speakers)

    labels: list[str | None] = [None] * len(segments)
    for i, c in zip(embedded_idx, cluster):
        labels[i] = f"C{c}"
    return _finalise(segments, labels)


def _merge_small_clusters(x: np.ndarray, cluster: np.ndarray, segs: list[dict], max_speakers: int) -> np.ndarray:
    """Fold tiny clusters (noise, laughs, one-off mis-clusterings) into the nearest real speaker."""
    durations = np.array([s["end"] - s["start"] for s in segs])
    share = {c: durations[cluster == c].sum() / durations.sum() for c in set(cluster)}
    keep = sorted((c for c in share if share[c] >= MIN_SPEAKER_SHARE), key=lambda c: -share[c])[:max_speakers]
    if not keep:
        return np.zeros_like(cluster)
    centroids = np.stack([x[cluster == c].mean(0) for c in keep])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    nearest = np.argmax(x @ centroids.T, axis=1)
    return np.array([c if c in keep else keep[nearest[i]] for i, c in enumerate(cluster)])


def _finalise(segments: list[dict], labels: list[str | None]) -> list[str]:
    """Fill unlabeled (very short) lines from the nearest labeled line; rename to S1, S2, ... by first appearance."""
    known = [i for i, lab in enumerate(labels) if lab is not None]
    if not known:
        return ["S1"] * len(segments)
    filled = []
    for i, lab in enumerate(labels):
        if lab is None:
            j = min(known, key=lambda k: abs(segments[k]["start"] - segments[i]["start"]))
            lab = labels[j]
        filled.append(lab)
    names: dict[str, str] = {}
    for lab in filled:
        names.setdefault(lab, f"S{len(names) + 1}")
    return [names[lab] for lab in filled]
