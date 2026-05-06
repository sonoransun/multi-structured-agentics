"""Method 2: Synergy delta.

For each task:

    delta = score(integrated) - max(score(agents_only), score(skills_only))

If `delta > 0` integration produced something neither side alone could, on
that task. If it's flat across the suite, the architecture isn't earning its
keep — that's a useful negative result.

This method directly tests the project's central claim. The other methods
are descriptive; this one is adjudicative.
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
    deltas = []

    for task in tasks:
        scores: dict[str, float] = {}
        for mode_name, mode_fn in MODES.items():
            try:
                output, trace = mode_fn(task, llm)
                scores[mode_name] = score_output(output.get("output"), task.expected)
            except Exception:
                scores[mode_name] = 0.0
                trace = None
                output = {}
            if collector is not None:
                collector.record(
                    task_id=task.id,
                    kind=task.kind.value,
                    prompt=task.prompt,
                    mode=mode_name,
                    output=output.get("output") if isinstance(output, dict) else output,
                    score=scores[mode_name],
                    trace=trace,
                )

        baseline = max(scores.get("agents_only", 0.0), scores.get("skills_only", 0.0))
        integrated = scores.get("integrated", 0.0)
        delta = integrated - baseline
        deltas.append(
            {
                "task": task.id,
                "kind": task.kind.value,
                "agents_only": round(scores.get("agents_only", 0.0), 3),
                "skills_only": round(scores.get("skills_only", 0.0), 3),
                "integrated": round(integrated, 3),
                "delta": round(delta, 3),
            }
        )

    mean_delta = sum(d["delta"] for d in deltas) / len(deltas) if deltas else 0.0
    wins = sum(1 for d in deltas if d["delta"] > 0)

    return {
        "tasks": deltas,
        "mean_delta": round(mean_delta, 3),
        "integration_wins": wins,
        "n_tasks": len(deltas),
        "verdict": _verdict(mean_delta, wins, len(deltas)),
    }


def _verdict(mean_delta: float, wins: int, n: int) -> str:
    if n == 0:
        return "no tasks"
    if mean_delta > 0.05 and wins >= n / 2:
        return "integration adds value"
    if mean_delta < -0.05:
        return "integration HURTS — investigate orchestration overhead"
    return "no clear synergy on this suite"
