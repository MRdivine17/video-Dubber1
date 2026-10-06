"""Quick check that XTTS voice cloning works: clone a speaker reference and speak one line.

    .venv\\Scripts\\python scripts\\test_voice.py work\\<video id>\\<run>\\voices\\S1.wav
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber.config import configure_environment  # noqa: E402

configure_environment()

import soundfile as sf  # noqa: E402

from dubber.tts import _xtts_once  # noqa: E402


def main(reference: Path) -> None:
    from TTS.api import TTS

    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
    model = tts.synthesizer.tts_model
    print(f"XTTS loaded in {time.time() - t0:.1f}s")
    gpt, emb = model.get_conditioning_latents(audio_path=[str(reference)], max_ref_length=30,
                                              gpt_cond_len=30, gpt_cond_chunk_len=4)
    text = ("Nothing you chase will make you happy by itself. What matters is how you live "
            "while you are chasing it. That is what we are talking about today.")
    t0 = time.time()
    wav = _xtts_once(model, text, gpt, emb, temperature=0.65)
    took = time.time() - t0
    sr = model.config.audio.output_sample_rate
    out = Path(__file__).resolve().parent.parent / "work" / "voice_test.wav"
    sf.write(str(out), wav, sr)
    print(f"spoke {len(wav) / sr:.1f}s of audio in {took:.1f}s (real-time factor {took / (len(wav) / sr):.2f}) -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
