from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..core.llm import LLM
from ..core.trace import Trace


@dataclass
class SkillResult:
    output: Any
    ok: bool
    error: str | None = None


class Skill(ABC):
    """A structured operation with a fixed contract.

    Skills are the "structured intent" half of the project. Each skill:

    - Has a stable name (used by the router and as a tool name when an agent
      invokes it)
    - Accepts a typed input dict (validated implicitly by `run`)
    - Returns a `SkillResult` whose `output` matches a documented schema

    Skills should be deterministic-ish: same input -> same shape of output.
    They MAY call an LLM internally (most do), but the LLM is constrained by
    a tight system prompt and, where applicable, structured output schemas.
    """

    name: str = ""
    description: str = ""

    def __init__(self, llm: LLM):
        self.llm = llm

    @abstractmethod
    def run(self, inputs: dict[str, Any], trace: Trace) -> SkillResult: ...

    def as_tool_schema(self) -> dict:
        """JSON-schema tool definition for use by an agent."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
