from __future__ import annotations

from ..core.trace import Trace
from .base import Agent, AgentResult

PERSONA = (
    "You are a coding specialist. Produce minimal, correct Python code that "
    "solves the user's problem. Output only the code in a single fenced "
    "block, nothing else. No explanations."
)


class CoderAgent(Agent):
    """Handles code-generation tasks."""

    name = "coder"
    persona = PERSONA

    def handle(self, prompt: str, trace: Trace) -> AgentResult:
        span = trace.open(self.name, "agent")
        try:
            resp = self.llm.complete(prompt, system=PERSONA, max_tokens=4096)
            trace.close(span, resp.tokens_in, resp.tokens_out, note="code-gen")
            return AgentResult(output={"code": resp.text.strip()}, ok=True)
        except Exception as e:
            trace.close(span, note=f"error: {e}")
            return AgentResult(output=None, ok=False, error=str(e))
