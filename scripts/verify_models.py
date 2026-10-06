"""Check every downloaded model weight file against the sha256 published by its source.

Hugging Face publishes a sha256 for each large (LFS) file; Demucs checkpoints carry a sha256
prefix in their file name. Needs network access for the Hugging Face lookups.

    .venv\\Scripts\\python scripts\\verify_models.py
"""
import hashlib
import os
import sys
from pathlib import Path

os.environ["DUB_ONLINE"] = "1"   # this check talks to the Hub
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber.config import MODELS_DIR, configure_environment  # noqa: E402

configure_environment()

# (repo id, local folder holding that repo's files)
HF_REPOS = [
    ("Systran/faster-whisper-large-v3", MODELS_DIR / "whisper" / "models--Systran--faster-whisper-large-v3"),
    ("facebook/nllb-200-distilled-1.3B", MODELS_DIR / "huggingface" / "hub" / "models--facebook--nllb-200-distilled-1.3B"),
    ("ai4bharat/indictrans2-indic-en-1B", MODELS_DIR / "huggingface" / "hub" / "models--ai4bharat--indictrans2-indic-en-1B"),
    ("speechbrain/spkrec-ecapa-voxceleb", MODELS_DIR / "huggingface" / "hub" / "models--speechbrain--spkrec-ecapa-voxceleb"),
    ("coqui/XTTS-v2", MODELS_DIR / "tts"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    checked = bad = 0
    for repo, folder in HF_REPOS:
        if not folder.exists():
            print(f"skip   {repo} (not downloaded)")
            continue
        try:
            siblings = api.model_info(repo, files_metadata=True).siblings
        except Exception as exc:
            print(f"skip   {repo} (could not fetch hashes: {str(exc)[:80]})")
            continue
        expected = {s.rfilename.split("/")[-1]: s.lfs.sha256 for s in siblings if s.lfs}
        for path in folder.rglob("*"):
            if path.is_file() and path.name in expected and path.stat().st_size > 1_000_000:
                ok = sha256(path) == expected[path.name]
                checked += 1
                bad += not ok
                print(f"{'OK     ' if ok else 'CORRUPT'} {path.stat().st_size / 1e9:5.2f} GB  {repo}/{path.name}")
    for th in (MODELS_DIR / "torch").rglob("*.th"):   # demucs: <sig>-<sha256 prefix>.th
        prefix = th.stem.split("-")[-1]
        ok = sha256(th).startswith(prefix)
        checked += 1
        bad += not ok
        print(f"{'OK     ' if ok else 'CORRUPT'} {th.stat().st_size / 1e9:5.2f} GB  demucs/{th.name}")
    print(f"checked {checked} model files: " + ("ALL MATCH" if checked and not bad else f"{bad} CORRUPT"))


if __name__ == "__main__":
    main()
