"""Method: Backend Pareto.

Run the integrated mode across multiple backends (stub, ollama, claude, ...)
and compute the (cost_usd, score) Pareto frontier. Answers the practical
question: which backend earns its keep on the most expensive code path?

Cost basis is real USD from `trace.cost_usd` (populated by LLM facade via
pricing.estimate_cost). Stub/ollama/transformers all report 0.0, so when
only free backends are available the frontier collapses to whichever has
the highest mean score.
"""
from __future__ import annotations

import time
from typing import Any

from msa.backends import (
    ClaudeBackend,
    OllamaBackend,
    StubBackend,
    TransformersBackend,
)
from msa.core.llm import LLM
from msa.core.types import Task
from msa.modes import MODES
from msa.training import TraceCollector

from ._grader import score_output


_BACKEND_CTORS = {
    "stub": StubBackend,
    "claude": ClaudeBackend,
    "ollama": OllamaBackend,
    "transformers": TransformersBackend,
}


def run(
    tasks: list[Task],
    llm: LLM | None = None,
    collector: TraceCollector | None = None,
    backends: list[str] | None = None,
) -> dict[str, Any]:
    backends = backends or ["stub", "ollama", "claude"]
    integrated = MODES["integrated"]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for bname in backends:
        ctor = _BACKEND_CTORS.get(bname)
        if ctor is None:
            skipped.append({"backend": bname, "reason": "unknown backend"})
            continue
        try:
            backend = ctor()
        except Exception as e:
            skipped.append({"backend": bname, "reason": f"ctor error: {e}"})
            continue
        if not getattr(backend, "available", False):
            skipped.append({"backend": bname, "reason": "not available"})
            continue

        b_llm = LLM(backend=backend)
        scores: list[float] = []
        costs: list[float] = []
        latencies: list[float] = []
        n_done = 0

        for task in tasks:
            t0 = time.time()
            try:
                output, trace = integrated(task, b_llm)
                score = score_output(output.get("output"), task.expected)
                err = None
            except Exception as e:
                output, trace = {}, None
                score = 0.0
                err = str(e)

            latency = time.time() - t0
            usd = float(trace.cost_usd) if trace else 0.0
            scores.append(score)
            costs.append(usd)
            latencies.append(latency)
            if err is None:
                n_done += 1

            if collector is not None:
                collector.record(
                    task_id=task.id,
                    kind=task.kind.value,
                    prompt=task.prompt,
                    mode=f"integrated@{bname}",
                    output=output.get("output") if isinstance(output, dict) else output,
                    score=score,
                    trace=trace,
                    error=err,
                )

        n = max(1, len(scores))
        rows.append(
            {
                "backend": bname,
                "mean_score": round(sum(scores) / n, 4),
                "mean_cost_usd": round(sum(costs) / n, 6),
                "mean_latency": round(sum(latencies) / n, 3),
                "n_tasks_completed": n_done,
                "n_tasks": len(tasks),
            }
        )

    triples = [(r["backend"], r["mean_cost_usd"], r["mean_score"]) for r in rows]
    frontier_names = set(_pareto_front(triples))
    frontier = sorted(
        [r for r in rows if r["backend"] in frontier_names],
        key=lambda r: (r["mean_cost_usd"], -r["mean_score"]),
    )
    dominated = [r for r in rows if r["backend"] not in frontier_names]

    return {
        "rows": rows,
        "frontier": frontier,
        "dominated": dominated,
        "summary": {
            "n_backends_run": len(rows),
            "n_backends_skipped": len(skipped),
            "skipped": skipped,
            "n_tasks": len(tasks),
        },
    }


def _pareto_front(points: list[tuple[str, float, float]]) -> list[str]:
    """Non-dominated names. (c_i, s_i) is dominated iff some (c_j, s_j) has
    c_j < c_i AND s_j > s_i (strict on both). Equal-cost peers don't
    dominate each other — needed for stub-only runs where every cost is 0.
    """
    out: list[str] = []
    for name, c, s in points:
        dominated = any(
            (c2 < c and s2 > s) for (n2, c2, s2) in points if n2 != name
        )
        if not dominated:
            out.append(name)
    return out
