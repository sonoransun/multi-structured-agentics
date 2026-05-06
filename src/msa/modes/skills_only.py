from __future__ import annotations

from ..core.llm import LLM
from ..core.trace import Trace
from ..core.types import Task
from ..skills import REGISTRY as SKILL_REGISTRY


def run(task: Task, llm: LLM) -> tuple[dict, Trace]:
    """Force every task through a single skill. No agent.

    For OPEN_ENDED tasks this is expected to underperform — the test of
    whether structured ops alone can carry a non-structured task.
    """
    trace = Trace()
    p = task.prompt.lower()

    if "extract" in p or "json" in p or "schema" in p:
        skill_name = "extract_json"
    else:
        skill_name = "summarize"

    skill = SKILL_REGISTRY[skill_name](llm)
    inputs = {"text": task.prompt, **task.inputs}
    res = skill.run(inputs, trace)
    return {"output": res.output, "ok": res.ok, "skill": skill_name}, trace
