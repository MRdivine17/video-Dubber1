"""Step 5 — translate every line into natural, speakable English.

Two passes:
1. Translation on the GPU.
   - Indian languages: Whisper translates each line's audio directly (dubber/speech_translate.py).
     Measured on a 99-minute Tamil podcast, recognise-then-translate dropped or garbled 14% of
     lines, mostly code-mixed English words written in Tamil script; translating from the
     audio keeps them. Lines where that comes back empty fall back to text translation.
   - Other languages: NLLB-200 on the transcript (IndicTrans2 for Indian languages if chosen).
2. Optional LLM polish (any provider, see dubber/llm.py): rewrites each draft for meaning and
   natural spoken phrasing, reading the source line and its neighbours, and keeps it inside a
   word budget derived from the time the original speaker took. That budget is what keeps
   the dub in sync without having to speed the voice up much.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import llm
from .gpu import DEVICE
from .media import free_gpu, read_json, write_json
from .reporting import ProgressLike, log
from .speech_translate import looks_unreliable, translate_speech

# Whisper language code -> FLORES-200 code.
INDIC = {
    "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym",
    "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru", "or": "ory_Orya",
    "as": "asm_Beng", "ur": "urd_Arab", "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Arab",
}
NLLB = {
    **INDIC,
    "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "nl": "nld_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "pl": "pol_Latn", "cs": "ces_Latn",
    "tr": "tur_Latn", "ar": "arb_Arab", "fa": "pes_Arab", "he": "heb_Hebr", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "zh": "zho_Hans", "id": "ind_Latn", "ms": "zsm_Latn", "vi": "vie_Latn",
    "th": "tha_Thai", "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn",
    "el": "ell_Grek", "hu": "hun_Latn", "ro": "ron_Latn", "bg": "bul_Cyrl", "sr": "srp_Cyrl",
    "hr": "hrv_Latn", "sk": "slk_Latn", "tl": "tgl_Latn", "sw": "swh_Latn", "si": "sin_Sinh",
}
LANGUAGE_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "tr": "Turkish",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.?!।॥。？！])\s+")

INDICTRANS_MODEL = "ai4bharat/indictrans2-indic-en-1B"
NLLB_MODEL = "facebook/nllb-200-distilled-1.3B"
BATCH = 16          # sentences per GPU batch: larger batches spill an 8 GB card into system RAM
NUM_BEAMS = 3


def _generation_args(inputs) -> dict:
    """Beam search bounded by the input length; no_repeat stops rare runaway repetition loops."""
    budget = min(256, int(inputs["input_ids"].shape[1] * 2) + 16)
    return {"num_beams": NUM_BEAMS, "max_new_tokens": budget, "no_repeat_ngram_size": 4}

# English speech runs at roughly 2.5-3 words per second; we budget 2.9 so lines fit at ~1.1x at worst.
WORDS_PER_SECOND = 2.9


def translate_segments(segments: list[dict], language: str, progress: ProgressLike, polish: bool,
                       llm_model: str | None, work_dir: Path, audio_16k: Path, whisper_model: str,
                       translator: str = "auto") -> tuple[list[str], list[str], dict]:
    """Return (machine drafts, final English lines, {engine, polish_model}), one line per segment.

    translator:
      "speech" - Whisper translates each line's audio directly to English
      "text"   - IndicTrans2 / NLLB-200 translate the recognised text
      "auto"   - speech for Indian languages (code-mixed speech loses content through text MT),
                 text for everything else
    `llm_model` overrides the auto-detected polish model (see dubber/llm.py); None = auto.
    Finished batches are saved under `work_dir`, so a stopped run resumes where it left off.
    """
    sources = [s["text"] for s in segments]
    checkpoints = {name: work_dir / f"{name}.partial.json"
                   for name in ("speech", "indictrans2", "nllb", "fallback", "polish")}
    use_speech = translator == "speech" or (translator == "auto" and language in INDIC)

    # Translation models first, so their GPU memory is freed before a local LLM is loaded.
    if language == "en":
        drafts, engine = sources, "none (already English)"
    elif use_speech:
        drafts = translate_speech(audio_16k, segments, language, whisper_model, progress, checkpoints["speech"])
        engine = f"Whisper {whisper_model} speech translation"
        weak = [i for i, (text, seg) in enumerate(zip(drafts, segments))
                if looks_unreliable(text, seg["end"] - seg["start"])]
        if weak and (language in NLLB):
            fixed, text_engine = _text_translate([sources[i] for i in weak], language, progress,
                                                 checkpoints["fallback"], checkpoints["fallback"], quiet=True)
            log(f"{len(weak)} of {len(segments)} lines came back empty or looping from speech translation; "
                f"those lines were translated from the transcript with {text_engine}")
            for i, text in zip(weak, fixed):
                if text:
                    drafts[i] = text
            engine += f" (+ {text_engine} on {len(weak)} lines)"
    elif language in NLLB:
        drafts, engine = _text_translate(sources, language, progress, checkpoints["indictrans2"], checkpoints["nllb"])
    else:
        drafts, engine = [""] * len(sources), "LLM only"   # no MT model: the LLM translates from source alone

    llm_cfg = _llm_ready(llm_model) if (polish or engine == "LLM only") and language != "en" else None
    if engine == "LLM only" and llm_cfg is None:
        raise ValueError(f"No translation model for language '{language}'. Configure an LLM (see .env.example).")

    finals = drafts
    if llm_cfg:
        try:
            finals = _polish(segments, drafts, language, llm_cfg, progress, checkpoints["polish"])
        finally:
            llm.release(llm_cfg)   # a local LLM must not keep VRAM the voice-cloning step needs
    for checkpoint in checkpoints.values():
        checkpoint.unlink(missing_ok=True)
    return drafts, finals, {"engine": engine, "polish_model": llm_cfg.describe() if llm_cfg else None}


def _text_translate(texts: list[str], language: str, progress: ProgressLike,
                    indictrans_checkpoint: Path, nllb_checkpoint: Path, quiet: bool = False) -> tuple[list[str], str]:
    """Text MT: IndicTrans2 for Indian languages when available, NLLB-200 otherwise."""
    if language in INDIC and indictrans_available(quiet):
        try:
            return _indictrans(texts, INDIC[language], progress, indictrans_checkpoint), "IndicTrans2-1B"
        except Exception as exc:
            log(f"IndicTrans2 failed ({exc}); falling back to NLLB-200", "warn")
    return _nllb(texts, NLLB[language], progress, nllb_checkpoint), "NLLB-200-1.3B"


# ── machine translation ─────────────────────────────────────────────────────────

def indictrans_available(quiet: bool = False) -> bool:
    """IndicTrans2 is gated on Hugging Face: usable once cached locally or when HF_TOKEN is set."""
    from huggingface_hub import try_to_load_from_cache

    cached = isinstance(try_to_load_from_cache(INDICTRANS_MODEL, "config.json"), str)
    if cached:
        return True
    if quiet:
        pass
    elif not os.environ.get("HF_TOKEN"):
        log("IndicTrans2 is gated (set HF_TOKEN after accepting its terms); using NLLB-200", "warn")
    else:
        # Dubbing runs with models offline; the gated download happens separately.
        log("IndicTrans2 not downloaded yet (run scripts/get_indictrans2.py once access is granted); "
            "using NLLB-200", "warn")
    return False


def _batched_generate(texts: list[str], translate_batch, progress: ProgressLike, label: str,
                      checkpoint: Path) -> list[str]:
    """Translate sentence by sentence, then rejoin each line.

    MT models silently drop content when one input holds several sentences, so lines are
    split at sentence punctuation first. Sentences are batched by length for GPU efficiency.
    """
    pieces, owner = [], []
    for i, text in enumerate(texts):
        for sentence in SENTENCE_SPLIT.split(text.strip()):
            if sentence.strip():
                pieces.append(sentence.strip())
                owner.append(i)

    done: dict[str, str] = read_json(checkpoint) if checkpoint.exists() else {}
    if done:
        log(f"Resuming translation: {len(done)} of {len(pieces)} sentences already translated")
    order = sorted((j for j in range(len(pieces)) if str(j) not in done), key=lambda j: len(pieces[j]))
    task = progress.add_task(label, total=len(pieces))
    progress.advance(task, len(done))
    for k in range(0, len(order), BATCH):
        idx = order[k:k + BATCH]
        for j, text in zip(idx, translate_batch([pieces[j] for j in idx])):
            done[str(j)] = text.strip()
        write_json(checkpoint, done)
        progress.advance(task, len(idx))
    translated = [done.get(str(j), "") for j in range(len(pieces))]

    lines: list[list[str]] = [[] for _ in texts]
    for j, i in enumerate(owner):
        lines[i].append(translated[j])
    return [" ".join(parts) for parts in lines]


def _indictrans(texts: list[str], src_code: str, progress: ProgressLike, checkpoint: Path) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError:
        from IndicTransToolkit import IndicProcessor

    device = DEVICE
    loading = progress.add_task("load IndicTrans2 onto the GPU", total=None)
    tokenizer = AutoTokenizer.from_pretrained(INDICTRANS_MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS_MODEL, trust_remote_code=True,
        dtype=torch.float16,
    ).to(device).eval()
    processor = IndicProcessor(inference=True)
    progress.update(loading, total=1, completed=1, description="IndicTrans2 ready on the GPU")

    def run(batch: list[str]) -> list[str]:
        prepped = processor.preprocess_batch(batch, src_lang=src_code, tgt_lang="eng_Latn")
        inputs = tokenizer(prepped, truncation=True, padding="longest", max_length=256,
                           return_tensors="pt", return_attention_mask=True).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, min_length=0, use_cache=True, **_generation_args(inputs))
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return processor.postprocess_batch(decoded, lang="eng_Latn")

    try:
        return _batched_generate(texts, run, progress, "translate (IndicTrans2)", checkpoint)
    finally:
        del model
        free_gpu()


def _nllb(texts: list[str], src_code: str, progress: ProgressLike, checkpoint: Path) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = DEVICE
    loading = progress.add_task("load NLLB-200 onto the GPU", total=None)
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang=src_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, dtype=torch.float16,
    ).to(device).eval()
    english = tokenizer.convert_tokens_to_ids("eng_Latn")
    progress.update(loading, total=1, completed=1, description="NLLB-200 ready on the GPU")

    def run(batch: list[str]) -> list[str]:
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, forced_bos_token_id=english, **_generation_args(inputs))
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    try:
        return _batched_generate(texts, run, progress, "translate (NLLB-200)", checkpoint)
    finally:
        del model
        free_gpu()


# ── LLM polish ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional film dubbing translator. You receive transcribed {language} speech, \
line by line, with a machine-translation draft for each line. Write the English line a voice actor will speak \
over the original video.

Rules:
- Translate the meaning, tone and intent of the source, not word for word. Sound like natural spoken English.
- The draft is only a hint. Read the source and fix anything the draft got wrong. If the draft is empty, translate the source.
- Each line must be speakable in its time slot: never use more than max_words words. Shorten by dropping filler, never meaning.
- Keep names, numbers and technical terms. No notes, brackets, stage directions or explanations.
- Lines are consecutive speech; keep references and pronouns consistent with the surrounding lines.
- If a source line is only a sound or filler (e.g. "hmm", "uh"), return a short natural equivalent.

Reply with JSON only: {{"lines": [{{"id": <id>, "en": "<english line>"}}]}} — exactly one entry per input id."""

LLM_BATCH = 30
LLM_WORKERS = 8


POLISH_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "en": {"type": "string"}},
                "required": ["id", "en"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def _llm_ready(model_override: str | None) -> llm.LLMConfig | None:
    """Detect the configured LLM and make one tiny request, so a bad key costs one warning, not a failed run."""
    cfg = llm.detect()
    if cfg is None:
        log("No LLM configured (no API key or local model server found): skipping the polish pass", "warn")
        return None
    cfg = llm.with_model(cfg, model_override)
    ok, detail = llm.check(cfg)
    if not ok:
        log(f"LLM polish disabled: {detail}", "warn")
        return None
    log(f"LLM polish: {detail}")
    return cfg


def word_budget(segments: list[dict], i: int) -> int:
    """Words that fit between this line's start and the next line's start (at most 1.5s past its own end)."""
    seg = segments[i]
    next_start = segments[i + 1]["start"] if i + 1 < len(segments) else seg["end"] + 1.5
    seconds = max(seg["end"] - seg["start"], min(next_start, seg["end"] + 1.5) - seg["start"])
    return max(2, int(seconds * WORDS_PER_SECOND))


def _polish(segments: list[dict], drafts: list[str], language: str, cfg: llm.LLMConfig,
            progress: ProgressLike, checkpoint: Path) -> list[str]:
    system = SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, language))
    finals = list(drafts)
    saved: dict[str, str] = read_json(checkpoint) if checkpoint.exists() else {}
    for i, text in saved.items():
        finals[int(i)] = text
    all_batches = [list(range(k, min(k + LLM_BATCH, len(segments)))) for k in range(0, len(segments), LLM_BATCH)]
    batches = [b for b in all_batches if not all(str(i) in saved for i in b)]
    task = progress.add_task(f"polish ({cfg.describe()})", total=len(segments))
    progress.advance(task, len(segments) - sum(len(b) for b in batches))
    workers = 2 if cfg.local else LLM_WORKERS   # a local model server handles few requests at once

    def run(idx: list[int]) -> dict[int, str]:
        context = [{"source": segments[j]["text"], "draft": drafts[j]} for j in range(max(0, idx[0] - 3), idx[0])]
        payload = {
            "previous_lines_for_context": context,
            "lines": [{"id": i, "source": segments[i]["text"], "draft": drafts[i], "max_words": word_budget(segments, i)}
                      for i in idx],
        }
        for attempt in range(3):
            try:
                answer = llm.complete_json(cfg, system, json.dumps(payload, ensure_ascii=False), POLISH_SCHEMA)
                lines = json.loads(answer)["lines"]
                result = {int(line["id"]): str(line["en"]).strip() for line in lines if str(line.get("en", "")).strip()}
                if all(i in result for i in idx) or attempt == 2:
                    return result
            except Exception as exc:
                if attempt == 2:
                    log(f"polish batch {idx[0]}-{idx[-1]} failed ({exc}); keeping drafts", "warn")
        return {}

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(run, idx): idx for idx in batches}
        for future in as_completed(futures):
            for i, text in future.result().items():
                if 0 <= i < len(finals):
                    finals[i] = text
                    saved[str(i)] = text
            write_json(checkpoint, saved)
            progress.advance(task, len(futures[future]))

    # Anything still empty (unsupported language + failed batch) falls back to the source text.
    return [f or d or segments[i]["text"] for i, (f, d) in enumerate(zip(finals, drafts))]
