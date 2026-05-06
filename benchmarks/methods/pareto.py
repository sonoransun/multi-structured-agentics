"""Method: cost-quality Pareto.

For every (task, mode) pair, record (cost, score). Aggregate per mode,
then compute the 2D non-dominated frontier: a mode is dominated iff some
other mode has BOTH lower cost AND higher score. When all costs are
equal (e.g. stub backend → cost_usd=0 everywhere) no mode is strictly
dominated on cost, so the frontier collapses to those that aren't
strictly score-dominated.

Cost basis: real USD when the trace carries it; otherwise a token proxy
(`0.5*tokens_in + 1.5*tokens_out`) so stub-mode runs still produce a
shape-correct result.
"""
from __future__ import annotations

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
    by_mode_cost: dict[str, list[float]] = {m: [] for m in MODES}
    by_mode_score: dict[str, list[float]] = {m: [] for m in MODES}
    saw_usd = False

    for task in tasks:
        for mode_name, mode_fn in MODES.items():
            try:
                output, trace = mode_fn(task, llm)
                score = score_output(output.get("output"), task.expected)
                err = None
            except Exception as e:
                output, trace = {}, None
                score = 0.0
                err = str(e)

            usd = float(trace.cost_usd) if trace else 0.0
            if usd > 0:
                saw_usd = True
                cost = usd
            else:
                tin = trace.tokens_in if trace else 0
                tout = trace.tokens_out if trace else 0
                cost = 0.5 * tin + 1.5 * tout

            by_mode_cost[mode_name].append(cost)
            by_mode_score[mode_name].append(score)

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

    points: list[dict[str, Any]] = []
    triples: list[tuple[str, float, float]] = []
    for mode_name in MODES:
        costs = by_mode_cost[mode_name]
        scores = by_mode_score[mode_name]
        if not costs:
            continue
        mean_cost = sum(costs) / len(costs)
        mean_score = sum(scores) / len(scores)
        points.append(
            {"mode": mode_name, "cost": round(mean_cost, 6), "score": round(mean_score, 4)}
        )
        triples.append((mode_name, mean_cost, mean_score))

    frontier_modes = set(_pareto_front(triples))
    frontier = sorted(
        [p for p in points if p["mode"] in frontier_modes],
        key=lambda p: (p["cost"], -p["score"]),
    )
    dominated = [p for p in points if p["mode"] not in frontier_modes]

    return {
        "points": points,
        "frontier": frontier,
        "dominated": dominated,
        "summary": {"cost_basis": "usd" if saw_usd else "tokens"},
    }


def _pareto_front(points: list[tuple[str, float, float]]) -> list[str]:
    """Return names of non-dominated points.

    A point (c_i, s_i) is dominated iff some (c_j, s_j) has c_j < c_i AND
    s_j > s_i (strict on both). Equal-cost or equal-score peers do not
    dominate each other — important for stub mode where every cost is 0.
    """
    out: list[str] = []
    for name, c, s in points:
        dominated = any(
            (c2 < c and s2 > s) for (n2, c2, s2) in points if n2 != name
        )
        if not dominated:
            out.append(name)
    return out
