from __future__ import annotations

from ..agents import REGISTRY as AGENT_REGISTRY
from ..core.llm import LLM
from ..core.trace import Trace
from ..core.types import Task, TaskKind


def run(task: Task, llm: LLM) -> tuple[dict, Trace]:
    """Force every task through an agent. Skills are not provided.

    For STRUCTURED tasks this is expected to produce looser outputs than the
    skill path — the test of whether agents alone are good enough.
    """
    trace = Trace()
    if task.kind == TaskKind.STRUCTURED or "code" not in task.prompt.lower():
        agent = AGENT_REGISTRY["researcher"](llm, skills=[])
    else:
        agent = AGENT_REGISTRY["coder"](llm, skills=[])

    res = agent.handle(task.prompt, trace)
    return {"output": res.output, "ok": res.ok, "agent": agent.name}, trace
