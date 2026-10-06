"""Compare text translation (NLLB / IndicTrans2) with speech translation on a finished run.

    python scripts\\compare_translation.py work\\<video id>\\<run>
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber.config import DEFAULT_WHISPER_MODEL, configure_environment  # noqa: E402

configure_environment()

import numpy as np  # noqa: E402

from dubber.reporting import CliReporter, set_reporter  # noqa: E402
from dubber.speech_translate import looks_unreliable, translate_speech  # noqa: E402


def ratio(text: str, source: str) -> float:
    return len(text.split()) / max(1, len(source.split()))


def main(work: Path) -> None:
    reporter = CliReporter()
    set_reporter(reporter)
    transcript = json.loads((work / "transcript.json").read_text(encoding="utf-8"))
    rows = json.loads((work / "translation.json").read_text(encoding="utf-8"))
    segments = transcript["segments"]
    with reporter.progress() as progress:
        speech = translate_speech(work / "vocals_16k.wav", segments, transcript["language"],
                                  DEFAULT_WHISPER_MODEL, progress, work / "speech_translation.json")

    text_r = np.array([ratio(r["english"], r["source"]) for r in rows if len(r["source"].split()) >= 4])
    speech_r = np.array([ratio(s, r["source"]) for s, r in zip(speech, rows) if len(r["source"].split()) >= 4])
    unreliable = sum(looks_unreliable(s, r["end"] - r["start"]) for s, r in zip(speech, rows))
    print(f"\ntext MT   : median words ratio {np.median(text_r):.2f}, lines < 0.9: {np.mean(text_r < 0.9) * 100:.0f}%")
    print(f"speech MT : median words ratio {np.median(speech_r):.2f}, lines < 0.9: {np.mean(speech_r < 0.9) * 100:.0f}%,"
          f" unreliable lines: {unreliable}")
    picks = [i for i, r in enumerate(rows) if r["start"] in (1553.0, 2790.8, 4354.6, 5381.8)]
    random.seed(3)
    picks += random.sample(range(len(rows)), 6)
    for i in picks:
        print(f"\n{rows[i]['start']:7.1f}s  TEXT  : {rows[i]['english'][:150]}")
        print(f"          SPEECH: {speech[i][:150]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
