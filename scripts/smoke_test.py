"""Quick environment check: imports, CUDA, and the library APIs the pipeline relies on."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber.config import configure_environment  # noqa: E402

configure_environment()

import torch  # noqa: E402

print("torch", torch.__version__, "cuda", torch.cuda.is_available())

import faster_whisper  # noqa: E402
from faster_whisper import BatchedInferencePipeline  # noqa: E402

params = inspect.signature(BatchedInferencePipeline.transcribe).parameters
print("faster-whisper", faster_whisper.__version__, "| batched params ok:",
      all(p in params for p in ("language", "batch_size", "beam_size", "word_timestamps", "vad_filter")))

import demucs  # noqa: E402
from demucs.apply import apply_model  # noqa: E402,F401

print("demucs", demucs.__version__)

import transformers  # noqa: E402

print("transformers", transformers.__version__)

try:
    from IndicTransToolkit.processor import IndicProcessor  # noqa: F401
    print("IndicTransToolkit ok (processor)")
except ImportError:
    from IndicTransToolkit import IndicProcessor  # noqa: F401
    print("IndicTransToolkit ok (top-level)")

from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402

cfg = XttsConfig()
print("xtts cfg:", cfg.gpt_cond_len, cfg.gpt_cond_chunk_len, cfg.max_ref_len, cfg.sound_norm_refs)
from TTS.tts.models.xtts import Xtts  # noqa: E402

print("xtts get_conditioning_latents:", list(inspect.signature(Xtts.get_conditioning_latents).parameters)[1:])

import speechbrain  # noqa: E402
from speechbrain.utils.fetching import LocalStrategy  # noqa: E402,F401

print("speechbrain", speechbrain.__version__)

import yt_dlp  # noqa: E402

print("yt-dlp", yt_dlp.version.__version__, "| js_runtimes param:",
      "js_runtimes" in inspect.getsource(yt_dlp.YoutubeDL.__init__) or "js_runtimes" in inspect.getsource(yt_dlp.YoutubeDL))

import edge_tts  # noqa: E402,F401
import openai  # noqa: E402

print("edge-tts ok | openai", openai.__version__)

from huggingface_hub import model_info  # noqa: E402

for repo in ("ai4bharat/indictrans2-indic-en-1B", "facebook/nllb-200-distilled-1.3B", "coqui/XTTS-v2"):
    try:
        print(repo, "gated:", model_info(repo).gated)
    except Exception as exc:
        print(repo, "ERR", str(exc)[:120])
