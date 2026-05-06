"""Three execution strategies for benchmarking.

Each mode exposes the same `run(task, llm) -> (output, trace)` interface so
benchmark methods can compare them apples-to-apples.

- agents_only: route every task to an agent; no skills available.
- skills_only: route every task to skills; no agent reasoning.
- integrated: full orchestrator (router + skills + agent).

The thesis under test: integrated > max(agents_only, skills_only) on at
least some task types. If not, the architecture isn't earning its keep.
"""
from .agents_only import run as run_agents_only
from .skills_only import run as run_skills_only
from .integrated import run as run_integrated

MODES = {
    "agents_only": run_agents_only,
    "skills_only": run_skills_only,
    "integrated": run_integrated,
}

__all__ = ["MODES", "run_agents_only", "run_skills_only", "run_integrated"]
