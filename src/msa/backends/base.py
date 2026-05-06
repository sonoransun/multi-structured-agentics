from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Response:
    text: str
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int = 0
    backend: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""


class Backend(ABC):
    """Pluggable LLM provider.

    Backends are interchangeable from the caller's perspective — same
    `complete()` signature, same `Response` shape. The caller (LLM facade,
    or a multi-model policy) picks which backend handles each call. This is
    what makes "local + Claude in one system" tractable: skills/agents don't
    know which backend they're on.

    Subclasses should set `name` and `available` (a class-level check that
    can run without raising — e.g. `True` if the import succeeds, else
    `False`). The facade uses `available` to fall back gracefully.

    Optional params (`tools`, `stream_callback`) are no-ops in stub /
    transformers / ollama and live in claude only. Default values preserve
    every existing caller's behavior.
    """

    name: str = "abstract"
    available: bool = False

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        cache_system: bool = True,
        tools: list[dict] | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> Response: ...
