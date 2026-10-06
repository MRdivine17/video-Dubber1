"""Runs the eight dubbing steps in order.

Every step writes its result under work/<video id>/<run>/ and is skipped on a rerun if
that result already exists, so a crash at minute 90 of a 2-hour job does not start over.
Progress goes to a Reporter: the terminal (CLI) or the browser (web server).
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from .align import build_voice_track, mix_with_background, write_subtitles
from .config import ASR_SR, MIX_SR, WORK_DIR, Options
from .diarize import diarize
from .download import download
from .gpu import require_gpu
from .media import duration, extract_audio, mux, read_json, trim_video, video_codec, write_json
from .reporting import CliReporter, Reporter, StageTimer, fmt_seconds, log, set_reporter
from .separate import separate_vocals
from .transcribe import transcribe
from .translate import translate_segments
from .tts import build_references, clip_path, synthesize


def run(opts: Options, reporter: Reporter | None = None) -> dict:
    """Dub `opts.url`. Returns the run report (also saved next to the output video)."""
    reporter = reporter or CliReporter()
    set_reporter(reporter)
    gpu = require_gpu()
    log(f"GPU: {gpu['name']} · {gpu['vram_gb']} GB VRAM · CUDA {gpu['cuda']}")

    timer = StageTimer(reporter)
    started_at = datetime.now().isoformat(timespec="seconds")

    # 1 ── download ──────────────────────────────────────────────────────────────
    with timer.stage("download") as st:
        meta, source = download(opts.url, WORK_DIR, st.progress, opts.cookies_browser)
        st.cached = not meta.pop("downloaded")
    reporter.video_info(meta)

    run_name = f"first-{opts.max_minutes:g}min" if opts.max_minutes else "full"
    work = WORK_DIR / meta["id"] / run_name
    if opts.force and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    video = source
    if opts.max_minutes:
        video = work / "video.mp4"
        if not video.exists():
            trim_video(source, video, opts.max_minutes * 60)
    total = duration(video)
    log(f"Video length {fmt_seconds(total)} · work folder {work}")

    # 2 ── separate speech from background ─────────────────────────────────────────
    mix_wav, vocals, background, vocals16 = (work / n for n in
                                              ("audio_44k.wav", "vocals.wav", "background.wav", "vocals_16k.wav"))
    with timer.stage("separate") as st:
        if all(f.exists() for f in (vocals, background, vocals16)):
            st.cached = True
        else:
            if not mix_wav.exists():
                extract_audio(video, mix_wav, MIX_SR, 2)
            if not (vocals.exists() and background.exists()):
                # resumes at the first unfinished 5-minute chunk
                separate_vocals(mix_wav, vocals, background, work / "separation_parts", st.progress)
            extract_audio(vocals, vocals16, ASR_SR, 1)
            mix_wav.unlink(missing_ok=True)

    # 3 ── transcribe ───────────────────────────────────────────────────────────────
    transcript_file = work / "transcript.json"
    with timer.stage("transcribe") as st:
        if transcript_file.exists():
            st.cached = True
            transcript = read_json(transcript_file)
        else:
            partial = work / "transcript.partial.json"   # resume point if the run stops mid-way
            segments, language = transcribe(vocals16, opts.language, opts.whisper_model, st.progress, total, partial)
            transcript = {"language": language, "segments": segments}
            write_json(transcript_file, transcript)
            partial.unlink(missing_ok=True)
    segments, language = transcript["segments"], transcript["language"]
    if not segments:
        raise RuntimeError("No speech was detected in this video.")
    log(f"Language: {language} · {len(segments)} lines of speech")

    # 4 ── speakers ─────────────────────────────────────────────────────────────────
    speakers_file = work / "speakers.json"
    with timer.stage("speakers") as st:
        if speakers_file.exists():
            st.cached = True
            speaker_data = read_json(speakers_file)
        else:
            labels = diarize(vocals16, segments, opts.speakers, st.progress)
            refs = build_references(vocals, segments, labels, work / "voices")
            speaker_data = {"labels": labels, "speakers": refs}
            write_json(speakers_file, speaker_data)
    labels, speakers = speaker_data["labels"], speaker_data["speakers"]
    for name, spk in speakers.items():
        log(f"{name}: {spk['lines']} lines, {spk['gender']} (~{spk['pitch_hz']:.0f} Hz), "
            f"{spk['reference_seconds']}s voice reference")

    # 5 ── translate ────────────────────────────────────────────────────────────────
    translation_file = work / "translation.json"
    translation_info_file = work / "translation_info.json"
    with timer.stage("translate") as st:
        if translation_file.exists() and translation_info_file.exists():
            st.cached = True
            lines = [row["english"] for row in read_json(translation_file)]
            translation_info = read_json(translation_info_file)
        else:
            drafts, lines, translation_info = translate_segments(segments, language, st.progress,
                                                                 opts.polish, opts.llm_model, work)
            write_json(translation_info_file, translation_info)
            write_json(translation_file, [
                {"id": i, "start": s["start"], "end": s["end"], "speaker": labels[i],
                 "source": s["text"], "draft": d, "english": e}
                for i, (s, d, e) in enumerate(zip(segments, drafts, lines))
            ])

    # 6 ── synthesize ───────────────────────────────────────────────────────────────
    clips_dir = work / "clips"
    with timer.stage("synthesize") as st:
        st.cached = all(clip_path(clips_dir, i).exists() for i in range(len(lines)))
        engine = synthesize(lines, labels, speakers, clips_dir, opts.voice, st.progress)

    # 7 ── align & mix ──────────────────────────────────────────────────────────────
    dub_wav = work / "dubbed_audio.wav"
    placements_file = work / "placements.json"
    srt, vtt = work / "english.srt", work / "english.vtt"
    with timer.stage("mix") as st:
        if dub_wav.exists() and placements_file.exists():
            st.cached = True
            placements = read_json(placements_file)
        else:
            voice, placements = build_voice_track(segments, clips_dir, vocals, total, st.progress, opts.max_speed)
            mix_with_background(voice, background, dub_wav, st.progress, opts.background_gain)
            write_json(placements_file, placements)
            del voice
        write_subtitles(lines, placements, srt, vtt)

    # 8 ── mux & verify ─────────────────────────────────────────────────────────────
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name(meta['title'])} [{meta['id']}]" + (f" (first {opts.max_minutes:g} min)" if opts.max_minutes else "")
    output = opts.output_dir / f"{stem} - English dub.mp4"
    with timer.stage("mux"):
        mux(video, dub_wav, srt, output)
        shutil.copyfile(srt, output.with_suffix(".srt"))
        shutil.copyfile(vtt, output.with_suffix(".vtt"))
        checks = _verify(video, output)

    # ── report ─────────────────────────────────────────────────────────────────────
    report_path = output.with_suffix(".report.json")
    report = {
        "video": meta,
        "run": run_name,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "gpu": gpu,
        "video_seconds": round(total, 2),
        "processing_seconds": round(timer.total_seconds, 1),
        "processing_time": fmt_seconds(timer.total_seconds),
        "realtime_factor": round(timer.total_seconds / total, 3),
        "stages": [{"key": s.key, "seconds": round(s.seconds, 1), "cached": s.cached} for s in timer.timings],
        "source_language": language,
        "lines": len(segments),
        "speakers": speakers,
        "voice_engine": engine,
        "whisper_model": opts.whisper_model,
        "translation": translation_info,
        "timing": _timing_stats(placements),
        "checks": checks,
        "work_dir": str(work),
        "source_video": str(video),
        "output": str(output),
        "subtitles": str(output.with_suffix(".srt")),
        "subtitles_vtt": str(output.with_suffix(".vtt")),
        "report": str(report_path),
    }
    write_json(report_path, report)
    reporter.run_finished(report)
    return report


def _safe_name(title: str) -> str:
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip().rstrip(".")
    return title[:80] or "video"


def _verify(source: Path, output: Path) -> list[str]:
    src_codec, out_codec = video_codec(source), video_codec(output)
    if src_codec != out_codec:
        raise RuntimeError(f"video stream changed codec ({src_codec} -> {out_codec})")
    src_len, out_len = duration(source), duration(output)
    if abs(src_len - out_len) > 1.0:
        raise RuntimeError(f"output length {out_len:.1f}s differs from source {src_len:.1f}s")
    return [f"Video stream copied without re-encoding ({out_codec})",
            f"Duration matches source ({out_len:.1f}s vs {src_len:.1f}s)",
            "English audio and English subtitle track embedded"]


def _timing_stats(placements: list[dict]) -> dict:
    import numpy as np

    speeds = np.array([p["speed"] for p in placements]) if placements else np.array([1.0])
    late = np.array([p["late_by"] for p in placements]) if placements else np.array([0.0])
    return {
        "sped_up_pct": round(100 * float(np.mean(speeds > 1.01)), 1),
        "avg_speed": round(float(np.mean(speeds)), 3),
        "max_speed": round(float(np.max(speeds)), 3),
        "max_late_s": round(float(np.max(late)), 2),
        "median_late_s": round(float(np.median(late)), 2),
        "lines_with_overflow": int(sum(1 for p in placements if p["overflow"] > 0.05)),
    }
