from __future__ import annotations

import os
from dataclasses import dataclass

from ..backends import (
    Backend,
    ClaudeBackend,
    OllamaBackend,
    Response,
    StubBackend,
    TransformersBackend,
)

DEFAULT_MODEL = "claude-opus-4-7"


@dataclass
class LLMResponse:
    """Backwards-compatible response shape."""
    text: str
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int = 0
    backend: str = ""


class LLM:
    """Facade over a `Backend`.

    Selection strategy (in order):

      1. Explicit `backend=` argument → use it (caller has decided)
      2. `MSA_BACKEND` env var → claude / ollama / transformers / stub
      3. Auto: ClaudeBackend if available, else OllamaBackend if Ollama is
         running locally, else StubBackend.

    The facade preserves the old API (`LLM().complete(...)` returns an
    `LLMResponse`) so existing skills/agents don't change. Multi-model
    routing — picking different backends for different roles — happens in
    `policy/multi_model.py`, which builds and holds multiple LLM instances.
    """

    def __init__(self, model: str = DEFAULT_MODEL, backend: Backend | None = None):
        self.model = model
        self.backend = backend or self._auto_select(model)

    @property
    def stub_mode(self) -> bool:
        return self.backend.name == "stub"

    def _auto_select(self, model: str) -> Backend:
        explicit = os.environ.get("MSA_BACKEND", "").lower()
        if explicit == "claude":
            b = ClaudeBackend(model=model)
            if b.available:
                return b
        if explicit == "ollama":
            b = OllamaBackend()
            if b.available:
                return b
        if explicit == "transformers":
            b = TransformersBackend()
            if b.available:
                return b
        if explicit == "stub":
            return StubBackend()

        # Auto fallback ladder
        for ctor in (lambda: ClaudeBackend(model=model), OllamaBackend):
            b = ctor()
            if b.available:
                return b
        return StubBackend()

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> LLMResponse:
        r: Response = self.backend.complete(
            prompt, system=system, max_tokens=max_tokens, cache_system=cache_system
        )
        return LLMResponse(
            text=r.text,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            cache_read_tokens=r.cache_read_tokens,
            backend=r.backend,
        )
