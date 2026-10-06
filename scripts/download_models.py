"""Pre-download every model the pipeline uses, retrying through flaky network / DNS.

    .venv\\Scripts\\python scripts\\download_models.py

After this, a dubbing run only needs the network for YouTube (and the optional LLM pass).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os  # noqa: E402

os.environ["DUB_ONLINE"] = "1"   # this script is the one place allowed to reach the model hubs
from dubber.config import MODELS_DIR, MODELS_READY_MARKER, configure_environment  # noqa: E402

configure_environment()


def demucs() -> None:
    from demucs.pretrained import get_model
    get_model("htdemucs")


def whisper() -> None:
    from faster_whisper import download_model
    from dubber.config import DEFAULT_WHISPER_MODEL
    download_model(DEFAULT_WHISPER_MODEL, cache_dir=str(MODELS_DIR / "whisper"))


def ecapa() -> None:
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                   savedir=str(MODELS_DIR / "speechbrain-ecapa"),
                                   run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY)


def indictrans2() -> None:
    from huggingface_hub import snapshot_download
    from dubber.translate import INDICTRANS_MODEL
    snapshot_download(INDICTRANS_MODEL)


def nllb() -> None:
    from huggingface_hub import snapshot_download
    from dubber.translate import NLLB_MODEL
    snapshot_download(NLLB_MODEL, allow_patterns=["*.json", "*.model", "*.bin", "*.safetensors", "*.txt"])


def xtts() -> None:
    from TTS.utils.manage import ModelManager
    ModelManager(progress_bar=True).download_model("tts_models/multilingual/multi-dataset/xtts_v2")


def with_retries(name: str, fn, attempts: int = 12) -> bool:
    for i in range(1, attempts + 1):
        try:
            t0 = time.time()
            fn()
            print(f"[ok] {name} ({time.time() - t0:.0f}s)", flush=True)
            return True
        except Exception as exc:
            print(f"[retry {i}/{attempts}] {name}: {str(exc)[:160]}", flush=True)
            time.sleep(min(30, 3 * i))
    print(f"[FAILED] {name}", flush=True)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    # NLLB is always fetched: it translates non-Indian languages and is the fallback for Indian ones.
    jobs = [("demucs htdemucs", demucs), ("whisper large-v3", whisper), ("speechbrain ECAPA", ecapa),
            ("XTTS v2", xtts), ("NLLB-200 1.3B", nllb)]
    # IndicTrans2 is a gated repo: it needs a Hugging Face token whose account accepted the model terms.
    if os.environ.get("HF_TOKEN"):
        jobs.append(("IndicTrans2 indic-en 1B", indictrans2))
    else:
        print("[skip] IndicTrans2 (gated on Hugging Face; set HF_TOKEN to enable). NLLB-200 is used instead.")
    results = [with_retries(name, fn) for name, fn in jobs]
    if all(results):
        MODELS_READY_MARKER.write_text("ok\n")   # dubbing runs now load models offline
        print("ALL_MODELS_OK", flush=True)
    else:
        print("SOME_MODELS_FAILED", flush=True)
