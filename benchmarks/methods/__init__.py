"""Benchmark methods.

Three methods are implemented; several others are documented in CLAUDE.md
under "Benchmark methods considered" but not yet built. They share a single
score function (graders/score_output) so cross-method comparisons are
meaningful.
"""
from .task_suite import run as run_task_suite
from .synergy import run as run_synergy
from .capability_matrix import run as run_capability_matrix

METHODS = {
    "task_suite": run_task_suite,
    "synergy": run_synergy,
    "capability_matrix": run_capability_matrix,
}

__all__ = ["METHODS", "run_task_suite", "run_synergy", "run_capability_matrix"]
