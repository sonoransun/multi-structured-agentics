"""Stub — implemented by T4 (Adversarial / counterfactual tasks)."""
from __future__ import annotations

from typing import Any

from msa.core.llm import LLM
from msa.core.types import Task
from msa.training import TraceCollector


def run(
    tasks: list[Task],
    llm: LLM | None = None,
    collector: TraceCollector | None = None,
) -> dict[str, Any]:
    return {"pairs": [], "mean_consistency": 0.0, "routing_sensitivity": 0.0, "note": "counterfactual stub — pending T4"}
