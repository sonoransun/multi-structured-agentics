from __future__ import annotations

from typing import Any

from ..core.trace import Trace
from .base import Skill, SkillResult

SYSTEM = (
    "You produce a single-sentence summary of the user's text. "
    "No preamble, no commentary, no quotes. Output the sentence and stop."
)


class SummarizeSkill(Skill):
    name = "summarize"
    description = "Produce a one-sentence summary of a passage."

    def run(self, inputs: dict[str, Any], trace: Trace) -> SkillResult:
        text = inputs.get("text", "")
        if not text:
            return SkillResult(output=None, ok=False, error="missing 'text'")

        span = trace.open(self.name, "skill")
        try:
            resp = self.llm.complete(text, system=SYSTEM, max_tokens=256)
            summary = resp.text.strip().split("\n")[0]
            trace.close(span, resp.tokens_in, resp.tokens_out, note="summarize", cost_usd=resp.cost_usd)
            return SkillResult(output=summary, ok=True)
        except Exception as e:
            trace.close(span, note=f"error: {e}")
            return SkillResult(output=None, ok=False, error=str(e))
