"""Method 3: Capability matrix.

Bins tasks by `TaskKind` (structured / open-ended / mixed) and reports each
mode's mean score per bin. The shape that should emerge from a healthy
architecture:

  - skills_only wins on STRUCTURED
  - agents_only wins on OPEN_ENDED
  - integrated matches or beats both on MIXED

Anything else is diagnostic. Integrated losing on its native bin (MIXED)
means orchestration overhead is exceeding its benefit; integrated dominating
everywhere may mean the simpler modes aren't well-tuned.
"""
from __future__ import annotations

from collections import defaultdict
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

    # bin -> mode -> [scores]
    grid: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for task in tasks:
        for mode_name, mode_fn in MODES.items():
            try:
                output, trace = mode_fn(task, llm)
                s = score_output(output.get("output"), task.expected)
            except Exception:
                s = 0.0
                output = {}
                trace = None
            grid[task.kind.value][mode_name].append(s)
            if collector is not None:
                collector.record(
                    task_id=task.id,
                    kind=task.kind.value,
                    prompt=task.prompt,
                    mode=mode_name,
                    output=output.get("output") if isinstance(output, dict) else output,
                    score=s,
                    trace=trace,
                )

    matrix = {
        kind: {mode: round(sum(ss) / len(ss), 3) for mode, ss in modes.items()}
        for kind, modes in grid.items()
    }

    return {"matrix": matrix, "interpretation": _interpret(matrix)}


def _interpret(matrix: dict[str, dict[str, float]]) -> list[str]:
    notes = []
    for kind, scores in matrix.items():
        if not scores:
            continue
        winner = max(scores.items(), key=lambda kv: kv[1])
        notes.append(f"{kind}: {winner[0]} wins ({winner[1]:.2f})")
    return notes
