"""Stub — implemented by T3 (Backend Pareto)."""
from __future__ import annotations

from typing import Any

from msa.core.llm import LLM
from msa.core.types import Task
from msa.training import TraceCollector


def run(
    tasks: list[Task],
    llm: LLM | None = None,
    collector: TraceCollector | None = None,
    backends: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "rows": [],
        "frontier": [],
        "dominated": [],
        "note": "backend_pareto stub — pending T3",
    }
