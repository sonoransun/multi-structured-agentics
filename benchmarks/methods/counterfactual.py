"""Method: Counterfactual / adversarial evaluation (Track T4).

Runs each ``COUNTERFACTUAL_PAIRS`` twin under every mode in ``MODES`` and
reports two complementary signals:

- **per-mode consistency** — ``1 - |score(a) - score(b)|`` averaged across
  pairs. A mode that's truly flexible should score similarly on twins. A
  mode that collapses (e.g. ``skills_only`` always emitting JSON) will look
  consistent only by being uniformly bad — that's why we also report…

- **routing sensitivity** — the fraction of pairs where the *winning* mode
  flips between ``a`` and ``b``. A healthy router/orchestrator should
  flip on most twins (that's the whole point of a minimal-edit pair). A
  router that picks the same mode for every input gets 0.0 here.

Both metrics together let us tell apart "mode is robust" from "mode is
indifferent". Stub mode produces degenerate scores; we treat ``score_a ==
score_b == 0`` as agreement (consistency = 1.0) so the harness still runs.
"""
from __future__ import annotations

from typing import Any

from msa.core.llm import LLM
from msa.core.types import Task
from msa.modes import MODES
from msa.training import TraceCollector

from ._grader import score_output


def run(
    tasks: list[Task],  # accepted for signature uniformity; pairs come from the module
    llm: LLM | None = None,
    collector: TraceCollector | None = None,
) -> dict[str, Any]:
    from benchmarks.adversarial_tasks import COUNTERFACTUAL_PAIRS

    llm = llm or LLM()

    pair_records: list[dict[str, Any]] = []
    consistency_by_mode: dict[str, list[float]] = {m: [] for m in MODES}
    routing_flips = 0

    for a, b in COUNTERFACTUAL_PAIRS:
        scores_a, scores_b = {}, {}
        for mode_name, mode_fn in MODES.items():
            scores_a[mode_name] = _run_and_score(mode_fn, a, llm, mode_name, collector)
            scores_b[mode_name] = _run_and_score(mode_fn, b, llm, mode_name, collector)

        per_mode: dict[str, dict[str, float]] = {}
        for mode_name in MODES:
            sa = scores_a[mode_name]
            sb = scores_b[mode_name]
            consistency = _consistency(sa, sb)
            consistency_by_mode[mode_name].append(consistency)
            per_mode[mode_name] = {
                "score_a": round(sa, 3),
                "score_b": round(sb, 3),
                "consistency": round(consistency, 3),
            }

        argmax_a = _argmax_mode(scores_a)
        argmax_b = _argmax_mode(scores_b)
        flip = 1 if argmax_a != argmax_b else 0
        routing_flips += flip

        pair_records.append(
            {
                "pair": [a.id, b.id],
                "argmax_a": argmax_a,
                "argmax_b": argmax_b,
                "routing_flipped": bool(flip),
                "by_mode": per_mode,
            }
        )

    n_pairs = len(COUNTERFACTUAL_PAIRS)
    mean_consistency = {
        mode: round(sum(xs) / len(xs), 3) if xs else 0.0
        for mode, xs in consistency_by_mode.items()
    }
    routing_sensitivity = round(routing_flips / n_pairs, 3) if n_pairs else 0.0

    return {
        "pairs": pair_records,
        "mean_consistency": mean_consistency,
        "routing_sensitivity": routing_sensitivity,
        "summary": {
            "n_pairs": n_pairs,
            "modes": list(MODES),
            "verdict": _verdict(mean_consistency, routing_sensitivity),
        },
    }


def _run_and_score(
    mode_fn,
    task: Task,
    llm: LLM,
    mode_name: str,
    collector: TraceCollector | None,
) -> float:
    try:
        output, trace = mode_fn(task, llm)
        score = score_output(output.get("output"), task.expected)
    except Exception:
        output, trace, score = {}, None, 0.0
    if collector is not None:
        collector.record(
            task_id=task.id,
            kind=task.kind.value,
            prompt=task.prompt,
            mode=mode_name,
            output=output.get("output") if isinstance(output, dict) else output,
            score=score,
            trace=trace,
        )
    return score


def _consistency(a: float, b: float) -> float:
    # Both modes failed → treat as agreement on failure (avoids
    # punishing stub mode). Otherwise: 1 - |Δ|.
    if a == 0.0 and b == 0.0:
        return 1.0
    return 1.0 - abs(a - b)


def _argmax_mode(scores: dict[str, float]) -> str:
    # Stable argmax: ties break in MODES order so the result is deterministic.
    best_mode = ""
    best_score = float("-inf")
    for mode in MODES:
        s = scores.get(mode, 0.0)
        if s > best_score:
            best_score = s
            best_mode = mode
    return best_mode


def _verdict(mean_consistency: dict[str, float], routing_sensitivity: float) -> str:
    # Mean across modes — a single readable number.
    if not mean_consistency:
        return "no pairs"
    avg = sum(mean_consistency.values()) / len(mean_consistency)
    if routing_sensitivity >= 0.5 and avg >= 0.5:
        return "router flips on twins; modes mostly consistent"
    if routing_sensitivity < 0.25:
        return "router insensitive to counterfactuals — investigate routing"
    if avg < 0.3:
        return "modes inconsistent across twins — composition is fragile"
    return "mixed signal — see per-pair table"
