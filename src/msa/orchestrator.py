from __future__ import annotations

from typing import Any

from .agents import REGISTRY as AGENT_REGISTRY
from .core.llm import LLM
from .core.trace import Trace
from .core.types import Task
from .policy import MultiModelPolicy, Role
from .routing import get_default_router
from .skills import REGISTRY as SKILL_REGISTRY


class Orchestrator:
    """Runs a task through the integrated pipeline.

    Two LLMs in flight: one for skills, one for agents. The policy decides
    which backend each role gets — see policy/multi_model.py. The
    orchestrator itself stays backend-agnostic.

    For the simple case (`Orchestrator(llm)`), both roles share the same
    LLM — preserves the original behavior. For the multi-model case,
    pass `policy=MultiModelPolicy(...)` instead.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        policy: MultiModelPolicy | None = None,
        router=None,
    ):
        if policy is not None:
            self.skill_llm = policy.llm_for(Role.SKILL)
            self.agent_llm = policy.llm_for(Role.AGENT)
            self.policy = policy
        else:
            self.skill_llm = llm or LLM()
            self.agent_llm = llm or self.skill_llm
            self.policy = None

        self.router = router or get_default_router()
        self.skills = {name: cls(self.skill_llm) for name, cls in SKILL_REGISTRY.items()}
        skill_list = list(self.skills.values())
        self.agents = {
            name: cls(self.agent_llm, skills=skill_list)
            for name, cls in AGENT_REGISTRY.items()
        }

    def run(self, task: Task, trace: Trace) -> dict[str, Any]:
        plan = self.router.route(task)
        router_span = trace.open("router", "router")
        trace.close(router_span, note=plan.rationale)

        skill_outputs: dict[str, Any] = {}
        for sname in plan.skills_first:
            skill = self.skills.get(sname)
            if not skill:
                continue
            inputs = {"text": task.prompt, **task.inputs}
            res = skill.run(inputs, trace)
            skill_outputs[sname] = res.output

        if plan.agent is None:
            final = next(iter(reversed(skill_outputs.values())), None)
            return {
                "output": final,
                "skill_outputs": skill_outputs,
                "agent": None,
                "plan": plan.rationale,
                "policy": self.policy.describe() if self.policy else None,
            }

        agent = self.agents[plan.agent]
        if skill_outputs:
            context = "\n\n".join(f"[{k}] {v}" for k, v in skill_outputs.items())
            agent_prompt = f"{task.prompt}\n\nContext from prior skills:\n{context}"
        else:
            agent_prompt = task.prompt

        ar = agent.handle(agent_prompt, trace)
        return {
            "output": ar.output,
            "skill_outputs": skill_outputs,
            "agent": plan.agent,
            "plan": plan.rationale,
            "ok": ar.ok,
            "policy": self.policy.describe() if self.policy else None,
        }
