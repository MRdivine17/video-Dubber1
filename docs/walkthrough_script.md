# 2-minute walkthrough script

Target: about 2:00 at a normal speaking pace (~280 words). Record your screen with the README pipeline diagram open, then show the two output videos.

---

**[0:00 – 0:15] What it is** *(show the terminal running `python dub.py <url>`)*

"This is my automated dubbing system. You give it a YouTube link in any language, and it produces the same video with English speech, in a clone of each original speaker's voice, with the original music and sound effects kept."

**[0:15 – 1:05] Architecture** *(show the pipeline diagram in the README)*

"It runs eight steps.
yt-dlp downloads the video.
Demucs separates the speech from the music and effects. That's the key to keeping the energy of the original: I only replace the voice, never the background.
Whisper large-v3 transcribes the speech with word-level timestamps, and I regroup the words into whole sentences, because translating half-sentences gives literal English.
Speaker embeddings, clustered, tell me who speaks each line.
IndicTrans2 translates Indian languages and NLLB handles the rest. An optional LLM pass rewrites each line for natural phrasing within a word budget based on how long the original speaker talked.
XTTS v2 clones each speaker from about twenty seconds of their own isolated voice.
Then every line is placed at its original timestamp, sped up only if it would run into the next line, matched to the original loudness, mixed back over the music, and ffmpeg swaps the audio track while copying the video stream untouched."

**[1:05 – 1:30] Decisions for long videos** *(show the work folder / report.json)*

"For a two-hour video, every step caches its result, so a crash resumes instead of restarting. Audio is processed in chunks, and models are loaded one at a time, so it fits in 8 GB of VRAM. Each run writes a report with the time per step and timing statistics."

**[1:30 – 2:00] Results** *(play 10 s of each dub, then show the timing table)*

"Here's the 30-minute dub. It took ___ to process. And the two-hour one took ___. Translation is meaning-based, the voice matches the speaker, and the lines stay in sync with the original."
