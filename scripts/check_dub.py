"""Round-trip check: transcribe the dubbed English audio with Whisper and compare it to the
English script, line by line. Low similarity flags garbled or hallucinated synthesis.

    .venv\\Scripts\\python scripts\\check_dub.py work\\<video id>\\<run>
"""
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber.config import DEFAULT_WHISPER_MODEL, MODELS_DIR, configure_environment  # noqa: E402

configure_environment()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def main(work: Path) -> None:
    import torch  # noqa: F401
    from faster_whisper import WhisperModel

    script = json.loads((work / "translation.json").read_text(encoding="utf-8"))
    placements = {p["id"]: p for p in json.loads((work / "placements.json").read_text(encoding="utf-8"))}
    model = WhisperModel(DEFAULT_WHISPER_MODEL, device="cuda", compute_type="float16",
                         download_root=str(MODELS_DIR / "whisper"))
    segments, _ = model.transcribe(str(work / "dubbed_audio.wav"), language="en", word_timestamps=True,
                                   vad_filter=True)
    heard = [(w.start, w.end, w.word) for s in segments for w in (s.words or [])]

    scores = []
    for row in script:
        p = placements[row["id"]]
        said = " ".join(w for a, b, w in heard if a >= p["start"] - 0.3 and b <= p["end"] + 0.3)
        ratio = difflib.SequenceMatcher(None, words(row["english"]), words(said)).ratio()
        scores.append(ratio)
        flag = "OK " if ratio >= 0.6 else "LOW"
        print(f"{flag} {ratio:.2f}  {p['start']:6.1f}s  script: {row['english'][:70]}")
        if ratio < 0.6:
            print(f"            heard:  {said[:70]}")
    print(f"\nmean word match {sum(scores) / len(scores):.2f} over {len(scores)} lines")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
