from __future__ import annotations

from ..core.llm import LLM
from ..core.trace import Trace
from ..core.types import Task
from ..orchestrator import Orchestrator


def run(task: Task, llm: LLM) -> tuple[dict, Trace]:
    """Full router + skills + agent pipeline."""
    trace = Trace()
    orch = Orchestrator(llm)
    out = orch.run(task, trace)
    return out, trace
