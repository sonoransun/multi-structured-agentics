from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..core.llm import LLM
from ..core.trace import Trace
from ..skills.base import Skill


@dataclass
class AgentResult:
    output: Any
    ok: bool
    error: str | None = None


class Agent(ABC):
    """An LLM-backed specialist for a domain.

    Agents are the "multi-agent specialty" half of the project. Each agent:

    - Has a persona encoded in its system prompt
    - Owns one task at a time and produces a single answer
    - May be given a set of `skills` to call as tools (the integration point)

    The agent base class deliberately does NOT implement a tool-use loop here.
    For the launch scaffold, agents either run plain text completion (when
    `skills` is empty) or invoke a single relevant skill in a hardcoded
    pre-step (composition pattern). A real tool-use loop is the natural next
    evolution — see CLAUDE.md's "Future work" section.
    """

    name: str = ""
    persona: str = ""

    def __init__(self, llm: LLM, skills: list[Skill] | None = None):
        self.llm = llm
        self.skills = {s.name: s for s in (skills or [])}

    @abstractmethod
    def handle(self, prompt: str, trace: Trace) -> AgentResult: ...
