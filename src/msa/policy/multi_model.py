"""Multi-model policy.

The thesis: not every step in a multi-skill / multi-agent pipeline needs
the same model. Skills are narrow, structured ops — a small local model is
often enough. Agents do open-ended reasoning — Claude is worth the cost.
The policy picks per-role.

Default policy:
    skill        → first available local backend (Ollama → stub)
    agent        → Claude if available, else local, else stub
    router       → never calls an LLM (keyword/embedding/learned)

Override via env:
    MSA_BACKEND_SKILL=ollama
    MSA_BACKEND_AGENT=claude

Or programmatically by passing `MultiModelPolicy(...)` into the
orchestrator (see orchestrator.py for how it's threaded).
"""
from __future__ import annotations

import os
from enum import Enum

from ..backends import (
    Backend,
    ClaudeBackend,
    OllamaBackend,
    StubBackend,
    TransformersBackend,
)
from ..core.llm import LLM


class Role(str, Enum):
    SKILL = "skill"
    AGENT = "agent"


def _build(name: str | None, prefer_local: bool) -> Backend:
    """Resolve a backend name to an instance, honoring availability."""
    candidates: list[type | Backend] = []
    if name == "claude":
        candidates = [ClaudeBackend]
    elif name == "ollama":
        candidates = [OllamaBackend]
    elif name == "transformers":
        candidates = [TransformersBackend]
    elif name == "stub":
        return StubBackend()
    else:
        candidates = (
            [OllamaBackend, ClaudeBackend] if prefer_local else [ClaudeBackend, OllamaBackend]
        )

    for c in candidates:
        b = c() if isinstance(c, type) else c
        if b.available:
            return b
    return StubBackend()


class MultiModelPolicy:
    """Holds one LLM per role and hands them out on demand."""

    def __init__(
        self,
        skill_backend: Backend | None = None,
        agent_backend: Backend | None = None,
    ):
        skill = skill_backend or _build(os.environ.get("MSA_BACKEND_SKILL"), prefer_local=True)
        agent = agent_backend or _build(os.environ.get("MSA_BACKEND_AGENT"), prefer_local=False)
        self._llms: dict[Role, LLM] = {
            Role.SKILL: LLM(backend=skill),
            Role.AGENT: LLM(backend=agent),
        }

    def llm_for(self, role: Role) -> LLM:
        return self._llms[role]

    def describe(self) -> dict[str, str]:
        return {role.value: llm.backend.name for role, llm in self._llms.items()}
