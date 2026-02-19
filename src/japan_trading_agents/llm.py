"""LLM provider abstraction using litellm."""

from __future__ import annotations

from typing import Any

import litellm
from loguru import logger

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Reasoning models only accept temperature=1 (they manage internal temperature themselves)
_REASONING_MODEL_PATTERNS = (
    "kimi-k2",          # Moonshot Kimi K2 series
    "kimi-thinking",    # Moonshot Kimi Thinking
    "o1", "o3",         # OpenAI o1/o3 family
    "deepseek-r1",      # DeepSeek R1
)


def _is_reasoning_model(model: str) -> bool:
    model_lower = model.lower()
    return any(pat in model_lower for pat in _REASONING_MODEL_PATTERNS)


class LLMClient:
    """Thin wrapper around litellm for multi-provider LLM access.

    Supports any model identifier that litellm understands:
      - "gpt-4o-mini", "gpt-4o"
      - "claude-sonnet-4-6", "claude-opus-4-6"
      - "moonshot/kimi-k2.5", "moonshot/kimi-latest"
      - "gemini/gemini-2.0-flash"
      - "ollama/llama3.2"

    Reasoning models (kimi-k2, o1, o3, deepseek-r1) automatically use
    temperature=1 regardless of the configured temperature value.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        # Reasoning models only accept temperature=1
        self.temperature = 1.0 if _is_reasoning_model(model) else temperature
        if _is_reasoning_model(model) and temperature != 1.0:
            logger.info(f"Reasoning model detected ({model}): using temperature=1.0")

    async def complete(self, system: str, user: str) -> str:
        """Run a single chat completion and return the content string."""
        logger.debug(f"LLM call: model={self.model}, system={system[:60]}...")
        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        content: str = response.choices[0].message.content or ""
        return content

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Run a completion expecting JSON output. Returns parsed dict."""
        import json

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw: str = response.choices[0].message.content or "{}"
        return json.loads(raw)  # type: ignore[no-any-return]
