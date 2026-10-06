"""OpenAI-compatible provider for the translation polish pass.

Covers OpenAI itself and every server that speaks the same API: Gemini's OpenAI endpoint,
and local model servers such as Ollama or LM Studio (base URL on localhost).
"""
from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI

from .llm import LLMConfig


@lru_cache(maxsize=4)
def _client(base_url: str | None, api_key_env: str | None, local: bool) -> OpenAI:
    api_key = os.environ.get(api_key_env or "", "") or ("local" if local else None)
    # Local models can be slow to load on first use; give them more time.
    return OpenAI(base_url=base_url, api_key=api_key, timeout=300 if local else 120, max_retries=2)


def complete_json(cfg: LLMConfig, system: str, user: str) -> str:
    response = _client(cfg.base_url, cfg.api_key_env, cfg.local).chat.completions.create(
        model=cfg.model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content or ""
