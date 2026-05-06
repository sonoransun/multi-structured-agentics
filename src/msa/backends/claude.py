from __future__ import annotations

import os
from typing import Any

from .base import Backend, Response

DEFAULT_MODEL = "claude-opus-4-7"


class ClaudeBackend(Backend):
    """Claude (Anthropic) via the official SDK.

    Uses adaptive thinking, streams to avoid HTTP timeouts on long outputs,
    and caches the system prompt via top-level `cache_control` on the last
    system block. Repeated calls with the same system prompt hit the cache;
    verify via `Response.cache_read_tokens`.

    `available` is False if the SDK isn't installed or the API key is missing
    — the LLM facade then falls back to the next backend in its preference
    list rather than raising at import time.
    """

    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client: Any = None
        self.available = False
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic()
            self.available = True
        except ImportError:
            pass

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> Response:
        if not self.available:
            raise RuntimeError("ClaudeBackend unavailable — no API key or SDK not installed")

        system_param: list[dict] | str = system
        if cache_system and system:
            system_param = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_param,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()

        text = next((b.text for b in msg.content if b.type == "text"), "")
        return Response(
            text=text,
            tokens_in=msg.usage.input_tokens,
            tokens_out=msg.usage.output_tokens,
            cache_read_tokens=getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
            backend=self.name,
        )
