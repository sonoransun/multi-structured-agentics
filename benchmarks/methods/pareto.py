"""Stub — implemented by T2 (Cost-quality Pareto)."""
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
    return {"points": [], "frontier": [], "dominated": [], "note": "pareto stub — pending T2"}
