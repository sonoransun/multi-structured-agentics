"""GSM8K loader.

Tries the public HuggingFace ``datasets`` package first. If that import or
download fails for any reason, falls back to a small bundled sample of
hand-written GSM8K-format word problems so the harness still has signal
in environments without network or the optional dep.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from msa.core.types import Task, TaskKind

_SAMPLE_PATH = Path(__file__).parent / "_gsm8k_sample.jsonl"
_ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def load(n: int = 20, split: str = "test") -> list[Task]:
    """Return up to ``n`` GSM8K Tasks.

    Each Task has ``kind=TaskKind.OPEN_ENDED`` and an ``expected`` contract
    of ``{"numeric_answer": "<final number>"}`` graded by ``_grader.py``.
    """
    rows = _load_rows(n=n, split=split)
    tasks: list[Task] = []
    for i, row in enumerate(rows[:n]):
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        final = _extract_final_number(answer)
        if not question or final is None:
            continue
        tasks.append(
            Task(
                id=f"gsm8k_{i}",
                prompt=f"{question}\n\nProvide the final numeric answer.",
                kind=TaskKind.OPEN_ENDED,
                expected={"numeric_answer": final},
            )
        )
    return tasks


def _load_rows(n: int, split: str) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("gsm8k", "main", split=split)
        rows: list[dict] = []
        for i, ex in enumerate(ds):
            if i >= n:
                break
            rows.append({"question": ex["question"], "answer": ex["answer"]})
        if rows:
            return rows
    except Exception:
        pass
    return _load_bundled()


def _load_bundled() -> list[dict]:
    if not _SAMPLE_PATH.exists():
        return []
    rows: list[dict] = []
    with _SAMPLE_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_final_number(answer: str) -> str | None:
    m = _ANSWER_RE.search(answer)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
    return nums[-1] if nums else None
