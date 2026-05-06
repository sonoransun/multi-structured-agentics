from __future__ import annotations

from ..core.trace import Trace
from .base import Agent, AgentResult

PERSONA = (
    "You are a tool-using assistant. You have access to a set of tools "
    "(structured skills). When a user request would benefit from a tool, "
    "call it; otherwise answer directly. Keep answers concise and grounded "
    "in tool outputs when tools were used."
)


class ToolUserAgent(Agent):
    """Agent that drives a real tool-use loop over its skills.

    Each iteration calls the LLM with `tools=` derived from `self.skills`.
    If the model emits tool_calls, we dispatch each to the matching skill
    and append the results back into the prompt, then loop. Otherwise we
    return the model's text.

    Stub mode (and any backend that doesn't emit tool_calls) doesn't drive
    the loop, so we fall back to a researcher-style keyword dispatch — that
    keeps the dep-free smoke tests honest about shape without pretending
    the stub understands tool schemas.
    """

    name = "tool_user"
    persona = PERSONA
    MAX_ITERS = 4

    def handle(self, prompt: str, trace: Trace) -> AgentResult:
        tools = [s.as_tool_schema() for s in self.skills.values()] or None
        current = prompt
        resp = None

        # Stub backends (and any backend without tool-use) won't emit
        # tool_calls. Skip the loop and use keyword fallback.
        if self.llm.stub_mode:
            return self._fallback(prompt, trace)

        try:
            for _ in range(self.MAX_ITERS):
                span = trace.open("tool_user_llm", "agent")
                resp = self.llm.complete(
                    current, system=self.persona, tools=tools, max_tokens=2048
                )
                trace.close(
                    span, resp.tokens_in, resp.tokens_out, note="tool_user"
                )

                if not resp.tool_calls:
                    return AgentResult(output=resp.text.strip(), ok=True)

                for tc in resp.tool_calls:
                    skill = self.skills.get(tc.name)
                    if skill is None:
                        current += f"\n\nTool result for {tc.name}: <unknown tool>"
                        continue
                    sr = skill.run(tc.arguments, trace)
                    current += f"\n\nTool result for {tc.name}: {sr.output}"

            text = (resp.text if resp else "").strip()
            return AgentResult(output=text, ok=True, error="max_iters")
        except Exception as e:
            return AgentResult(output=None, ok=False, error=str(e))

    def _fallback(self, prompt: str, trace: Trace) -> AgentResult:
        """Keyword dispatch for stub/no-tool backends.

        Mirrors researcher.py's pre-step pattern: if the prompt names a
        known skill, run it; otherwise fall through to a plain completion.
        """
        span = trace.open("tool_user_llm", "agent")
        resp = self.llm.complete(prompt, system=self.persona, max_tokens=2048)
        trace.close(span, resp.tokens_in, resp.tokens_out, note="tool_user_stub")
        text = resp.text.strip()

        lower = prompt.lower()
        if "summarize" in self.skills and "summarize" in lower:
            sr = self.skills["summarize"].run({"text": prompt}, trace)
            if sr.ok:
                return AgentResult(
                    output={"answer": text, "summary": sr.output}, ok=True
                )

        return AgentResult(output=text, ok=True)
