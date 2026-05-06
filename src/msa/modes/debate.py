"""Multi-agent debate mode.

Two LLM personas independently produce candidate answers; a judge call
either picks the better one or synthesizes them. The synthesis is the
mode's output. Falls back to candidate A if the judge can't produce
anything usable (notably in stub mode, where outputs are short hashes).
"""
from __future__ import annotations

from ..core.llm import LLM
from ..core.trace import Trace
from ..core.types import Task

PERSONA_A = (
    "You are a skeptical empiricist. Prefer concrete, verifiable claims; "
    "flag unsupported speculation; keep answers tight."
)
PERSONA_B = (
    "You are a creative theorist. Prefer broad framings and unexpected "
    "connections; surface possibilities the literal reading misses."
)
JUDGE_SYSTEM = (
    "You are an impartial judge. Read two candidate answers to the same "
    "task and either pick the stronger one verbatim or synthesize a "
    "better answer that combines their merits. Output only the final "
    "answer text — no preface, no commentary."
)

_TRUNC = 1000


def _call(llm: LLM, trace: Trace, name: str, prompt: str, system: str) -> str:
    span = trace.open(name, "agent")
    resp = llm.complete(prompt, system=system)
    trace.close(
        span,
        tokens_in=resp.tokens_in,
        tokens_out=resp.tokens_out,
        cost_usd=resp.cost_usd,
    )
    return resp.text or ""


def run(task: Task, llm: LLM) -> tuple[dict, Trace]:
    """Two personas debate; a judge synthesizes."""
    trace = Trace()

    a_text = _call(llm, trace, "debate_candidate_a", task.prompt, PERSONA_A)
    b_text = _call(llm, trace, "debate_candidate_b", task.prompt, PERSONA_B)

    judge_prompt = (
        f"TASK:\n{task.prompt}\n\n"
        f"CANDIDATE A (skeptical-empiricist):\n{a_text[:_TRUNC]}\n\n"
        f"CANDIDATE B (creative-theorist):\n{b_text[:_TRUNC]}\n\n"
        "Which is better and why? Or synthesize a better answer. "
        "Return only the final answer text."
    )
    judge_text = _call(llm, trace, "debate_judge", judge_prompt, JUDGE_SYSTEM)

    synth = judge_text.strip() if isinstance(judge_text, str) else ""
    if not synth:
        synth = a_text

    return (
        {
            "output": synth,
            "ok": True,
            "candidates": [a_text, b_text],
            "judge": judge_text,
            "mode": "debate",
        },
        trace,
    )
