"""LLM-judge grader (Track T1).

Asks the LLM to score an output on a structured rubric (correctness,
contract-adherence, conciseness) and parse a JSON response. Falls back
to the deterministic contract grader on any failure or in stub mode, so
tests stay reproducible.
"""
from __future__ import annotations

import json
import re
from typing import Any

from msa.core.llm import LLM

from ._grader import _score_contract

_RUBRIC = """You are an impartial grader. Score the candidate output for the given task on a 0-1 scale across three axes:

  - correctness: does it answer the prompt and arrive at a valid result?
  - contract: does it satisfy the structural expectations (keys, mentions, format)?
  - conciseness: is it tight and on-topic, with no padding?

Take the unweighted mean of the three axes as the final score. Respond with a single JSON object on one line and nothing else:

  {"score": <float in [0,1]>, "why": "<one short sentence>"}
"""


def judge(
    prompt: str,
    output: Any,
    expected: dict | None,
    llm: LLM,
) -> tuple[float, str]:
    """Return (score, rationale). Falls back to contract grader on failure."""
    if llm is None or llm.stub_mode:
        return _score_contract(output, expected), "fallback: contract"

    user = (
        f"TASK PROMPT:\n{prompt}\n\n"
        f"EXPECTED CONTRACT (may be partial):\n{json.dumps(expected, default=str)}\n\n"
        f"CANDIDATE OUTPUT:\n{json.dumps(output, default=str)[:4000]}\n\n"
        "Now score it."
    )
    try:
        resp = llm.complete(user, system=_RUBRIC, max_tokens=256)
        data = _parse_json(resp.text)
        score = float(data["score"])
        why = str(data.get("why", ""))[:280]
        score = max(0.0, min(1.0, score))
        return score, why
    except Exception:
        return _score_contract(output, expected), "fallback: contract"


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*?\"score\".*?\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object with score in response")
