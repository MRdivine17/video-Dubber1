# Automated Video Dubbing

Give it a YouTube URL in any language. It produces the same video with English speech in a clone of each original speaker's voice. The background music and sound effects stay, and the video stream is copied untouched.

```
python dub.py "https://www.youtube.com/watch?v=..."
```

The terminal shows a progress bar for every step and ends with a processing-time table. The `output/` folder gets:

| File | What it is |
|---|---|
| `<title> [<id>] - English dub.mp4` | Final video: original video stream, English audio, embedded English subtitles |
| `<title> [<id>] - English dub.srt` | English subtitles, timed to the dubbed speech |
| `<title> [<id>] - English dub.report.json` | Processing time per step, speakers found, timing stats, verification checks |

## Pipeline

```
YouTube URL
  1  yt-dlp ─────────────► source.mp4  (max 1080p; video stream is never re-encoded)
  2  Demucs htdemucs (GPU) ► vocals.wav        speech only
                           ► background.wav    music + effects, kept for the final mix
  3  faster-whisper large-v3 (GPU, batched, VAD)
        → source-language text with word timestamps → regrouped into sentences
  4  Speakers: ECAPA embeddings + clustering (or pyannote 3.1 if HF_TOKEN is set)
        → each line labelled S1, S2, ...; ~20 s voice reference cut per speaker
  5  Translate: Indian languages → Whisper translates each line's audio directly
        (text MT fallback on lines that come back empty); others → NLLB-200-1.3B
        → optional LLM polish: natural phrasing + a word budget per line
  6  Coqui XTTS v2: English speech cloned from each speaker's reference
        (edge-tts neural voices as fallback, or with --voice edge)
  7  Align: each line starts at the original time, is sped up (pitch kept) only
     if it overruns the next line, and is matched to the original loudness;
     mixed over background.wav
  8  ffmpeg: -c:v copy + new AAC audio + subtitle track → output .mp4, then verified
```

| Module | Step |
|---|---|
| `dub.py` | Command line and entry point |
| `dubber/pipeline.py` | Runs the steps in order, caching and the final report |
| `dubber/download.py` | 1: yt-dlp |
| `dubber/separate.py` | 2: Demucs, processed in 5-minute chunks |
| `dubber/transcribe.py` | 3: Whisper and sentence regrouping |
| `dubber/diarize.py` | 4: speaker detection |
| `dubber/translate.py` | 5: translator choice, IndicTrans2 / NLLB text MT, LLM polish |
| `dubber/speech_translate.py` | 5: Whisper speech-to-English translation per line |
| `dubber/tts.py` | 6: voice references, XTTS cloning, edge-tts |
| `dubber/align.py` | 7: time fitting, loudness matching, mix, SRT |
| `dubber/media.py` | ffmpeg / ffprobe helpers |
| `dubber/reporting.py` | `Reporter` interface: rich terminal output or web job state |
| `dubber/gpu.py` | CUDA-only guard and GPU info |
| `server/app.py`, `server/jobs.py` | FastAPI endpoints, GPU job queue, SSE progress |
| `web/` | React + TypeScript UI (Vite) |

## Design decisions

- **Separate the speech before doing anything else.** Replacing the whole soundtrack would also delete the music and effects, and the video would lose its energy. Demucs splits the speech from the background, so only the speech is dubbed. The isolated speech also gives Whisper cleaner input (fewer hallucinations over music) and clean voice references for cloning.
- **Sentences, not Whisper segments.** Whisper's segments often break a sentence in half. Translating half-sentences gives literal, broken English. Word timestamps let us rebuild whole sentences and keep their exact start and end times.
- **Translate from the audio for Indian languages.** I measured where words went missing on a 99-minute Tamil podcast (`scripts/diagnose_dub.py`):
  - Recognition missed 0.5% of the speech.
  - Voice synthesis reproduced 97% of the English words.
  - Text translation lost or garbled 14% of the lines.

  Indian-language speech mixes in English words ("content creation"). Whisper writes those words in Tamil script, then the text model mistranslates them ("Canton Cration Mall"). So for Indian languages, Whisper translates each transcript line's audio straight to English, on that line's exact time span, which keeps the timing. Lines that come back empty or looping fall back to text translation. `--translator text|speech` overrides the automatic choice.
- **Translate for meaning.** For other languages NLLB-200 translates the transcript; IndicTrans2 is used for Indian languages when text translation is chosen. When an LLM is available on the machine (any provider, detected automatically, see below), an LLM pass rewrites each draft. It reads the source line and the lines around it, fixes mistranslations, makes the English sound spoken rather than written, and keeps each line within a word budget based on how long the original speaker talked.
- **Timing is solved in two places.**
  1. The translation is asked to fit the time available.
  2. The aligner speeds a line up only when it would run into the next line. It keeps the pitch (ffmpeg rubberband) and caps the speed-up at 1.35x.

  If a line still overruns, the next line starts slightly late, and the delay is recovered at the next pause, so error never accumulates over a long video. Every line's speed and delay are recorded in `placements.json` and summarised in the report.
- **Same voice.** XTTS v2 clones each speaker from about 20 seconds of their own isolated speech. Speakers are found by clustering ECAPA voice embeddings, which needs no account. If `HF_TOKEN` is set, pyannote 3.1 is used instead.
- **Same energy.** Each dubbed line is scaled to the original speaker's loudness on that line, then mixed over the untouched background.
- **Same video.** `ffmpeg -c:v copy`. The pipeline checks the codec and duration of the output against the source and fails loudly if anything changed.
- **Built for 2-hour videos.**
  - Every step caches its result in `work/<video id>/`, and each voice clip is saved as soon as it is made. An interrupted run resumes at the step (or the clip) where it stopped.
  - Demucs runs in chunks and the final mix is streamed, so memory stays flat.
  - Models are loaded one step at a time and freed afterwards, so everything fits in 8 GB of VRAM.

## Web app (React UI + Python API)

```
.venv\Scripts\python serve.py          # API + built UI  ->  http://127.0.0.1:8000
```

1. Paste a link. The UI shows the video's thumbnail, title and length straight away.
2. Pick the options (full video or the first N minutes, cloned or neural voice, number of speakers, LLM polish) and start.
3. Watch it run:
   - an overall progress bar with an ETA;
   - all 8 steps, each with its own sub-progress bars and timings;
   - a live log, streamed from the GPU worker over Server-Sent Events.
4. When it finishes you get:
   - a player that switches between the English dub and the original at the same playhead, with subtitles;
   - downloads for the MP4, SRT and report;
   - timing quality numbers;
   - the cloned speakers, with their voice references playable;
   - a per-step time breakdown;
   - a searchable transcript (original and English side by side) where clicking a line jumps the video there.
5. Every finished dub stays in the library on the home page.

| Endpoint | Purpose |
|---|---|
| `GET /api/system` | GPU, model weights on disk, detected LLM (provider, model, local/cloud) |
| `POST /api/llm/refresh` | Re-detect API keys and local model servers |
| `GET /api/preview?url=` | Video metadata before dubbing |
| `POST /api/jobs` | Start a dub (`url`, `max_minutes`, `voice`, `speakers`, `polish`) |
| `GET /api/jobs/{id}/events` | Live job snapshots (SSE) |
| `POST /api/jobs/{id}/cancel` | Cancel; cached steps are kept, so a resume continues from them |
| `GET /api/library`, `GET /api/library/{report}/transcript` | Finished dubs and their line-by-line transcripts |
| `/media/output/*` | Output videos and subtitles (supports Range requests, so seeking works) |

Jobs run one at a time on a single GPU worker thread (`server/jobs.py`). The web server and `dub.py` call the same `dubber.pipeline.run()`. The only difference is the `Reporter` that receives the progress events: terminal bars for the CLI, job state for the browser.

Frontend development: `cd web && npm install && npm run dev` (http://localhost:5173, proxied to the API on :8000). After changes, run `npm run build` so `serve.py` picks them up.

## Setup (Windows, NVIDIA GPU)

```
powershell -ExecutionPolicy Bypass -File setup_env.ps1   # env + packages + model weights + UI build
start_studio.cmd                                          # web app -> http://127.0.0.1:8000
dub.cmd "https://www.youtube.com/watch?v=..."             # or the command line
```

`setup_env.ps1` installs everything listed in `requirements.txt` (pinned, CUDA PyTorch included) into `%USERPROFILE%\.venvs\video-dubber` on the system drive. Models, videos and outputs stay in this folder.

GPU only: every model runs on CUDA (Whisper int8/fp16, Demucs, NLLB/IndicTrans2 fp16, ECAPA, XTTS), with TF32 enabled. If no CUDA device is found, the run stops with a clear error instead of silently falling back to the CPU.

**Resuming:** a stopped or cancelled run continues where it stopped. Press Resume in the UI, or run the same command again.
- Finished steps are reused as they are.
- Inside a step, work continues from the last save point:
  - the yt-dlp download resumes its partial file;
  - speech/music separation, per 5-minute chunk;
  - transcription, from the last saved timestamp (saved every 15 s);
  - translation and the LLM polish, per batch;
  - voice synthesis, per line.
- A video already on disk is reused without contacting YouTube.

- Requires ffmpeg (with rubberband) and Node.js on PATH. yt-dlp uses Node to unlock YouTube formats.
- Model weights download on first use into `models/`.
- Optional: an LLM for the polish pass. Copy `.env.example` to `.env` and add whichever key you have, or just run a local model server. Nothing is tied to one account or model. Detection order (first match wins):
  1. `DUB_LLM_PROVIDER`, if set
  2. `ANTHROPIC_API_KEY` (Claude)
  3. `OPENAI_API_KEY`, plus `OPENAI_BASE_URL` / `OPENAI_API_BASE` for any OpenAI-compatible server. A localhost URL means a local model, such as Ollama or LM Studio.
  4. `GEMINI_API_KEY` / `GOOGLE_API_KEY`
  5. A running Ollama server

  The model is `DUB_LLM_MODEL` if set; otherwise it's discovered from the provider's model list. The web UI shows what was found and lets you pick a different model; the CLI takes `--llm-model`.
- Integrity check: `scripts\verify_install.py` re-hashes every installed package file against its wheel RECORD, which catches files silently damaged on disk. Repair with `scripts\repair_env.ps1 -Packages "name1,name2"`.
- Font: the UI uses **Anthropic Sans** when it's installed on the machine; otherwise it falls back to IBM Plex Sans, which is bundled. Anthropic Sans is proprietary, so it isn't bundled.
- Optional: set `HF_TOKEN` (after accepting the pyannote model terms) to use pyannote for speaker detection.

## Options

```
--max-minutes N       dub only the first N minutes (quick test)
--lang hi|ta|de|...   skip language auto-detection
--voice clone|edge    XTTS voice cloning (default) or edge-tts stock voices
--speakers N          force the number of speakers
--translator auto|speech|text   translate from the audio or the transcript (auto: audio for Indian languages)
--no-polish           skip the LLM pass
--force               ignore cached results (clean timing run)
```

## Licences

XTTS v2 weights are released under the Coqui Public Model License, which is non-commercial. The other models are open (MIT, Apache-2.0 or CC-BY-NC for NLLB).
