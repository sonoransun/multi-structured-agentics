from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskKind(str, Enum):
    """Coarse classification used by the router.

    `STRUCTURED` favors a skill (deterministic transform with a known schema).
    `OPEN_ENDED` favors an agent (model-driven exploration).
    `MIXED` requires both — the orchestrator's reason for existing.
    """

    STRUCTURED = "structured"
    OPEN_ENDED = "open_ended"
    MIXED = "mixed"


@dataclass
class Task:
    id: str
    prompt: str
    kind: TaskKind
    inputs: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] | None = None


@dataclass
class Result:
    task_id: str
    output: Any
    mode: str
    success: bool
    score: float
    latency_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None
