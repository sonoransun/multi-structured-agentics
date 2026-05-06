"""Shared grader. Returns a [0, 1] score per task.

The grader is intentionally simple — it checks the contract on the task's
`expected` field rather than calling an LLM judge. This makes every method
deterministic and free to run, which matters when the benchmark itself is
under development. Plug an LLM-judge grader in here when you're ready to
trade reproducibility for nuance.
"""
from __future__ import annotations

from typing import Any


def score_output(output: Any, expected: dict | None) -> float:
    """Public entry point. Layer T1's LLM judge on top of `_score_contract`
    rather than editing this body.
    """
    return _score_contract(output, expected)


def _score_contract(output: Any, expected: dict | None) -> float:
    if expected is None:
        return 1.0 if output is not None else 0.0

    score = 0.0
    checks = 0

    text_blob = _extract_text(output)

    if "required_keys" in expected:
        checks += 1
        data = _extract_dict(output)
        if data is not None:
            present = sum(1 for k in expected["required_keys"] if k in data)
            score += present / len(expected["required_keys"])

    if "must_mention" in expected:
        checks += 1
        hits = sum(1 for w in expected["must_mention"] if w.lower() in text_blob.lower())
        score += hits / len(expected["must_mention"])

    if "must_contain_code" in expected:
        checks += 1
        hits = sum(1 for w in expected["must_contain_code"] if w in text_blob)
        score += hits / len(expected["must_contain_code"])

    if "max_sentences" in expected:
        checks += 1
        n_sentences = max(1, text_blob.count(".") + text_blob.count("!") + text_blob.count("?"))
        score += 1.0 if n_sentences <= expected["max_sentences"] else 0.5

    if "numeric_answer" in expected:
        checks += 1
        score += _score_numeric(text_blob, expected["numeric_answer"])

    if "python_unit_test" in expected:
        checks += 1
        score += _score_python_unit_test(text_blob, expected["python_unit_test"])

    return score / checks if checks else (1.0 if output else 0.0)


def _score_numeric(text: str, expected_answer: str) -> float:
    import re

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return 0.0
    try:
        target = float(str(expected_answer).strip().replace(",", ""))
    except ValueError:
        return 0.0
    last = float(nums[-1])
    return 1.0 if abs(last - target) < 1e-6 else 0.0


def _score_python_unit_test(text: str, test_code: str) -> float:
    import re
    import subprocess
    import sys
    import tempfile
    import textwrap

    code_match = re.search(r"```(?:python)?\n(.+?)```", text, re.DOTALL)
    code = code_match.group(1) if code_match else text
    program = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code) + "\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(program)
            path = fh.name
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=5,
            text=True,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except (subprocess.TimeoutExpired, Exception):
        return 0.0


def _extract_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        # Common shapes from agents/skills/orchestrator
        for key in ("answer", "summary", "code", "output"):
            v = output.get(key)
            if isinstance(v, str):
                return v
        return str(output)
    return str(output)


def _extract_dict(output: Any) -> dict | None:
    if isinstance(output, dict):
        # Walk one level to find a JSON-ish payload
        if any(k in output for k in ("name", "role", "priority", "category")):
            return output
        for v in output.values():
            if isinstance(v, dict) and any(
                k in v for k in ("name", "role", "priority", "category")
            ):
                return v
    return None
