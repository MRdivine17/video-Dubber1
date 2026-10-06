"""Pick the LLM for the translation polish pass from whatever this machine has configured.

Nothing is hard-coded to one account or model. Detection order (first match wins):

1. DUB_LLM_PROVIDER = anthropic | openai | gemini | ollama | none   (explicit choice)
2. ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN                         -> Claude
3. OPENAI_API_KEY (+ OPENAI_BASE_URL / OPENAI_API_BASE if set)       -> OpenAI or any
   OpenAI-compatible server (a localhost base URL = a local model, e.g. Ollama / LM Studio)
4. GEMINI_API_KEY / GOOGLE_API_KEY                                   -> Gemini
5. A local Ollama server answering on OLLAMA_HOST (default localhost:11434)

The model comes from DUB_LLM_MODEL if set; otherwise it is discovered from the provider's
model list (local servers: the first installed model). Keys are also read from a `.env`
file in the project root, so anyone running this locally just drops their key there.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .config import ROOT

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Defaults used only when a provider can't list its models and DUB_LLM_MODEL is unset.
CLAUDE_DEFAULT = "claude-opus-5-5"
# For cloud OpenAI-compatible providers, prefer a fast model from what the account actually offers.
FAST_MODEL_HINTS = ("mini", "flash", "haiku", "small", "lite")


@dataclass
class LLMConfig:
    provider: str                 # anthropic | openai | gemini | ollama | openai-compatible
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    local: bool = False           # runs on this machine (no data leaves it)
    models: list[str] = field(default_factory=list)

    def describe(self) -> str:
        where = "local" if self.local else "cloud"
        return f"{self.provider} · {self.model} ({where})"

    def to_dict(self) -> dict:
        return asdict(self)


def load_dotenv_file() -> None:
    """Read KEY=value pairs from <project>/.env into the environment (real env vars win)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def _get_json(url: str, headers: dict | None = None, timeout: float = 4.0) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _is_local(url: str | None) -> bool:
    return bool(url) and any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))


def _list_openai_models(base_url: str, api_key: str) -> list[str]:
    data = _get_json(base_url.rstrip("/") + "/models", {"Authorization": f"Bearer {api_key}"})
    return [m["id"] for m in (data or {}).get("data", []) if "id" in m]


def _pick(models: list[str], prefer_fast: bool) -> str | None:
    if not models:
        return None
    chat = [m for m in models if not any(x in m for x in ("embed", "whisper", "tts", "image", "audio", "moderation",
                                                             "realtime", "transcribe", "search", "dall-e"))]
    pool = chat or models
    if prefer_fast:
        for hint in FAST_MODEL_HINTS:
            fast = sorted((m for m in pool if hint in m), reverse=True)
            if fast:
                return fast[0]
    return pool[0]


def _ollama_base() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/")


def _anthropic() -> LLMConfig:
    model = os.environ.get("DUB_LLM_MODEL") or os.environ.get("ANTHROPIC_MODEL") or CLAUDE_DEFAULT
    return LLMConfig("anthropic", model, api_key_env="ANTHROPIC_API_KEY")


def _openai_compatible() -> LLMConfig:
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    local = _is_local(base)
    models = _list_openai_models(base or "https://api.openai.com/v1", key)
    model = os.environ.get("DUB_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or _pick(models, prefer_fast=not local)
    provider = ("ollama" if base and "11434" in base else "openai-compatible") if base else "openai"
    return LLMConfig(provider, model or "", base_url=base, api_key_env="OPENAI_API_KEY", local=local, models=models)


def _gemini() -> LLMConfig:
    env = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY"
    models = [m.removeprefix("models/") for m in _list_openai_models(GEMINI_BASE, os.environ[env])]
    model = os.environ.get("DUB_LLM_MODEL") or _pick(models, prefer_fast=True)
    return LLMConfig("gemini", model or "", base_url=GEMINI_BASE, api_key_env=env, models=models)


def _ollama() -> LLMConfig | None:
    base = _ollama_base()
    tags = _get_json(base + "/api/tags", timeout=2.0)
    if tags is None:
        return None
    models = [m["name"] for m in tags.get("models", [])]
    model = os.environ.get("DUB_LLM_MODEL") or (models[0] if models else "")
    return LLMConfig("ollama", model, base_url=base + "/v1", local=True, models=models)


@lru_cache(maxsize=1)
def detect() -> LLMConfig | None:
    """The LLM this machine is set up for, or None. Cached for the process."""
    load_dotenv_file()
    choice = (os.environ.get("DUB_LLM_PROVIDER") or "").lower()
    if choice == "none":
        return None
    if choice == "anthropic" or (not choice and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))):
        return _anthropic()
    if choice in ("openai", "openai-compatible") or (not choice and os.environ.get("OPENAI_API_KEY")):
        return _openai_compatible()
    if choice == "gemini" or (not choice and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))):
        return _gemini()
    if choice in ("", "ollama"):
        return _ollama()
    return None


def with_model(cfg: LLMConfig, model: str | None) -> LLMConfig:
    """Same provider, a different model (user override from the CLI / UI)."""
    return cfg if not model else LLMConfig(**{**cfg.to_dict(), "model": model})


def complete_json(cfg: LLMConfig, system: str, user: str, schema: dict) -> str:
    """One request whose answer is a JSON document matching `schema`. Returns the JSON text."""
    if cfg.provider == "anthropic":
        from .llm_anthropic import complete_json as claude_json

        return claude_json(cfg.model, system, user, schema)
    from .llm_openai import complete_json as openai_json

    return openai_json(cfg, system, user)


def release(cfg: LLMConfig) -> None:
    """Free GPU memory held by a local Ollama model, so the voice-cloning step gets the whole GPU."""
    if cfg.provider != "ollama" or not cfg.base_url:
        return
    native = cfg.base_url.rstrip("/").removesuffix("/v1")
    body = json.dumps({"model": cfg.model, "keep_alive": 0}).encode()
    try:
        req = urllib.request.Request(native + "/api/generate", data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass   # best effort: the model also unloads by itself after Ollama's keep-alive timeout


def check(cfg: LLMConfig) -> tuple[bool, str]:
    """Tiny live request: is the provider reachable and the key/model accepted?"""
    if cfg.local and _get_json(cfg.base_url.rstrip("/") + "/models", timeout=2.0) is None:
        return False, f"{cfg.provider} at {cfg.base_url} is not running (start it, then re-detect)"
    if not cfg.model:
        return False, f"{cfg.provider}: no model found (set DUB_LLM_MODEL)"
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}
    try:
        complete_json(cfg, "Reply with JSON only.", 'Return {"ok": true}', schema)
        return True, cfg.describe()
    except Exception as exc:  # unreachable server, bad key, unknown model, ...
        hint = " (is the local server running?)" if cfg.local else ""
        return False, f"{cfg.describe()}: {str(exc)[:160]}{hint}"
