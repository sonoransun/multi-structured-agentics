"""Method: Judged suite (Track T1).

Mirrors task_suite but emits BOTH a contract score and an LLM-judge score
per row, plus their Pearson correlation. Useful for validating that the
deterministic grader tracks human-ish nuance — or exposing where it doesn't.
"""
from __future__ import annotations

import math
import time
from typing import Any

from msa.core.llm import LLM
from msa.core.types import Task
from msa.modes import MODES
from msa.training import TraceCollector

from ._grader import _score_contract, score_output
from ._judge import judge


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
                raw = output.get("output") if isinstance(output, dict) else output
                score_contract = score_output(raw, task.expected)
                score_judge, why = judge(task.prompt, raw, task.expected, llm)
                err = None
            except Exception as e:
                output, trace, raw = {}, None, None
                score_contract = 0.0
                score_judge = 0.0
                why = f"error: {e}"
                err = str(e)
            rows.append(
                {
                    "task": task.id,
                    "kind": task.kind.value,
                    "mode": mode_name,
                    "score_contract": round(score_contract, 3),
                    "score_judge": round(score_judge, 3),
                    "judge_why": why,
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
                    output=raw,
                    score=score_contract,
                    trace=trace,
                    error=err,
                    judge_score=score_judge,
                )

    return {"rows": rows, "summary": _summarize(rows)}


def _summarize(rows: list[dict]) -> dict[str, Any]:
    by_mode_c: dict[str, list[float]] = {}
    by_mode_j: dict[str, list[float]] = {}
    for r in rows:
        by_mode_c.setdefault(r["mode"], []).append(r["score_contract"])
        by_mode_j.setdefault(r["mode"], []).append(r["score_judge"])
    contracts = [r["score_contract"] for r in rows]
    judges = [r["score_judge"] for r in rows]
    return {
        "mean_contract": {m: round(sum(s) / len(s), 3) for m, s in by_mode_c.items()},
        "mean_judge": {m: round(sum(s) / len(s), 3) for m, s in by_mode_j.items()},
        "pearson_r": round(_pearson(contracts, judges), 3),
        "n": len(rows),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)
