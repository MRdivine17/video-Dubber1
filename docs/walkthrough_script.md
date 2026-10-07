# 2-minute walkthrough script

The brief asks the walkthrough to explain three things: **your dubs**, **your architecture**, and **your decisions**. It is graded on how clearly you explain them (code quality and clarity) and on how good the dubs are (accuracy).

About 280 words, so 2:00 at a calm pace. Fill the `[__]` blanks after the final clean runs.

---

**[0:00 – 0:15] What it is** *(screen: terminal running `dub.cmd "<url>"`, progress bars moving)*

"Hi, I'm [name]. This is my automated dubbing system. You give it a YouTube link in any language, and it returns the same video in English, in each speaker's own cloned voice, with the original music kept and the video untouched."

**[0:15 – 0:40] The dubs** *(play ~8 s of the original Tamil podcast, then the same 8 s dubbed)*

"Here's a Tamil podcast with two speakers. *[original]* And the dub. *[dub]* Each voice is cloned separately, and the English lands where the original lines did. The 30-minute video took [__] to process, and the [2-hour] one took [__], on a single 8 GB RTX 3060 Ti."

**[0:40 – 1:05] Architecture** *(screen: pipeline diagram in the README)*

"It's eight steps, one module each. yt-dlp downloads. Demucs splits the speech from the music. Whisper large-v3 transcribes with word timestamps, and I rebuild full sentences, because half-sentences translate badly. Voice embeddings group the lines by speaker. Then translation, XTTS cloning from about twenty seconds of each speaker, alignment and mixing, and ffmpeg swaps the audio while copying the video stream untouched."

**[1:05 – 1:40] Decisions** *(screen: "Design decisions" in the README, or the `diagnose_dub.py` output)*

"Three decisions matter most. First, separate the speech before anything else: I replace only the voice, so the energy of the background stays. Second, I measured where quality was lost. Text translation garbled fourteen percent of lines, because speakers mix in English words. So for Indian languages, Whisper translates each line directly from the audio. Third, timing: a line is only sped up if it would overlap the next one, capped at 1.35x with the pitch kept. Three in four lines play at natural speed."

**[1:40 – 2:00] Long videos and code** *(screen: `work/` folder, `report.json`, then the web app)*

"For long videos, every step caches its result and resumes after a crash, and models load one at a time to fit in 8 GB. The same pipeline runs from the terminal or this web app, and every run writes a report with timings and checks. Thanks for watching."

---

## Recording tips

- Cut the clips before recording; don't wait on a live run. For the opening shot, a short cached run (`--max-minutes 1`) shows the progress bars without waiting.
- Pick an 8-second stretch where both speakers talk and the translation reads well (check the `.srt`).
- Speak slower than feels natural; 2:00 is a limit, not a target. Cut the last section's middle sentence first if you run long.
