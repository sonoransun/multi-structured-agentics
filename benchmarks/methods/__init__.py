"""Benchmark methods.

Each method is registered in METHODS by name. They share a single score
function (graders/score_output) so cross-method comparisons are meaningful.
"""
from .task_suite import run as run_task_suite
from .synergy import run as run_synergy
from .capability_matrix import run as run_capability_matrix
from .judged_suite import run as run_judged_suite
from .pareto import run as run_pareto
from .backend_pareto import run as run_backend_pareto
from .counterfactual import run as run_counterfactual

METHODS = {
    "task_suite": run_task_suite,
    "synergy": run_synergy,
    "capability_matrix": run_capability_matrix,
    "judged_suite": run_judged_suite,
    "pareto": run_pareto,
    "backend_pareto": run_backend_pareto,
    "counterfactual": run_counterfactual,
}

__all__ = [
    "METHODS",
    "run_task_suite",
    "run_synergy",
    "run_capability_matrix",
    "run_judged_suite",
    "run_pareto",
    "run_backend_pareto",
    "run_counterfactual",
]
