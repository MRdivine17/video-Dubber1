"""Download the gated IndicTrans2 model (needs HF_TOKEN in .env and accepted model terms).
Retries through network / DNS drops. Doesn't import the audio stack, so it runs anywhere.
"""
import os
import sys
import time
from pathlib import Path

os.environ["DUB_ONLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubber.config import configure_environment  # noqa: E402

configure_environment()

from huggingface_hub import auth_check, snapshot_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError  # noqa: E402

REPO = "ai4bharat/indictrans2-indic-en-1B"

for attempt in range(1, 21):
    try:
        auth_check(REPO)
        t0 = time.time()
        path = snapshot_download(REPO)
        files = {p.name: p.stat().st_size for p in Path(path).rglob("*") if p.is_file()}
        weights = sum(size for name, size in files.items() if name.endswith((".bin", ".safetensors")))
        print(f"[ok] {len(files)} files, {weights / 1e9:.2f} GB of weights, {time.time() - t0:.0f}s", flush=True)
        sys.exit(0 if weights > 1e9 else 2)
    except GatedRepoError:
        print("[denied] access not granted yet for this token's account", flush=True)
        sys.exit(3)
    except Exception as exc:
        print(f"[retry {attempt}/20] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        time.sleep(min(30, 3 * attempt))
sys.exit(1)
