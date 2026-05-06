"""Method 1: Fixed task suite.

Runs every (task, mode) pair and reports a flat table. The simplest signal:
which mode wins overall, by how much, at what cost.
"""
from __future__ import annotations

import time
from typing import Any

from msa.core.llm import LLM
from msa.core.types import Task
from msa.modes import MODES
from msa.training import TraceCollector

from ._grader import score_output


def run(
    tasks: list[Task],
    llm: LLM | None = None,
    collector: TraceCollector | None = None,
) -> dict[str, Any]:
    llm = llm or LLM()
    rows = []

    for task in tasks:
        for mode_name, mode_fn in MODES.items():
            t0 = time.time()
            try:
                output, trace = mode_fn(task, llm)
                score = score_output(output.get("output"), task.expected)
                err = None
            except Exception as e:
                output, trace = {}, None
                score = 0.0
                err = str(e)
            rows.append(
                {
                    "task": task.id,
                    "kind": task.kind.value,
                    "mode": mode_name,
                    "score": round(score, 3),
                    "latency_s": round(time.time() - t0, 3),
                    "tokens_in": trace.tokens_in if trace else 0,
                    "tokens_out": trace.tokens_out if trace else 0,
                    "error": err,
                }
            )
            if collector is not None:
                collector.record(
                    task_id=task.id,
                    kind=task.kind.value,
                    prompt=task.prompt,
                    mode=mode_name,
                    output=output.get("output") if isinstance(output, dict) else output,
                    score=score,
                    trace=trace,
                    error=err,
                )

    summary = _summarize(rows)
    return {"rows": rows, "summary": summary}


def _summarize(rows: list[dict]) -> dict[str, Any]:
    by_mode: dict[str, list[float]] = {}
    cost_by_mode: dict[str, list[int]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r["score"])
        cost_by_mode.setdefault(r["mode"], []).append(r["tokens_in"] + r["tokens_out"])
    return {
        "mean_score": {m: round(sum(s) / len(s), 3) for m, s in by_mode.items()},
        "mean_tokens": {m: round(sum(c) / len(c), 1) for m, c in cost_by_mode.items()},
    }
