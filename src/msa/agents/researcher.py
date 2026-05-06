from __future__ import annotations

from ..core.trace import Trace
from .base import Agent, AgentResult

PERSONA = (
    "You are a research specialist. Given an open-ended question, produce a "
    "concise, factual answer grounded in widely-known information. Cite "
    "uncertainty when present. Prefer accuracy over breadth."
)


class ResearcherAgent(Agent):
    """Handles open-ended factual / analytical questions.

    If a `summarize` skill is available, the agent uses it to produce a
    one-line digest in addition to the full answer — a small demonstration
    of agent-uses-skill composition.
    """

    name = "researcher"
    persona = PERSONA

    def handle(self, prompt: str, trace: Trace) -> AgentResult:
        span = trace.open(self.name, "agent")
        try:
            resp = self.llm.complete(prompt, system=PERSONA, max_tokens=2048)
            answer = resp.text.strip()
            trace.close(span, resp.tokens_in, resp.tokens_out, note="research")

            if "summarize" in self.skills:
                sk = self.skills["summarize"]
                sr = sk.run({"text": answer}, trace)
                if sr.ok:
                    return AgentResult(
                        output={"answer": answer, "summary": sr.output}, ok=True
                    )

            return AgentResult(output={"answer": answer}, ok=True)
        except Exception as e:
            trace.close(span, note=f"error: {e}")
            return AgentResult(output=None, ok=False, error=str(e))
