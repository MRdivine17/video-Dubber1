"""Claude provider for the translation polish pass (Anthropic Python SDK).

Credentials resolve the SDK's standard way: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
`ant auth login` profile. Answers are constrained to the caller's JSON schema with
structured outputs, so they always parse.
"""
from __future__ import annotations

import anthropic

# Models that take output_config.effort and the server-side refusal fallback.
_CURRENT_GEN = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=3)
    return _client


def complete_json(model: str, system: str, user: str, schema: dict) -> str:
    output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
    current = model.startswith(_CURRENT_GEN)
    if current:
        output_config["effort"] = "low"   # short rewriting task: little thinking needed

    if current:
        # On a policy decline, the API re-runs the request on Anthropic's recommended fallback model.
        response = _get_client().beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        )
    else:
        response = _get_client().messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )

    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise RuntimeError(f"Claude declined this batch (category: {category})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer was cut off at max_tokens")
    return "".join(block.text for block in response.content if block.type == "text")
