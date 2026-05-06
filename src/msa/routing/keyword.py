from __future__ import annotations

from ..core.types import Task, TaskKind
from .base import Route, Router


class KeywordRouter(Router):
    """Heuristic router based on TaskKind + keyword matching.

    Identical to the launch scaffold's `router.py` logic — kept transparent
    and free. The router class wraps the pure function so it slots into the
    Router protocol used by the policy layer.
    """

    available = True

    def route(self, task: Task) -> Route:
        p = task.prompt.lower()

        if task.kind == TaskKind.STRUCTURED:
            if "extract" in p or "schema" in p or "json" in p:
                return Route(["extract_json"], None, "structured extraction → skill only")
            if "summarize" in p or "summary" in p:
                return Route(["summarize"], None, "summarization → skill only")
            return Route(["summarize"], None, "default structured → summarize")

        if task.kind == TaskKind.OPEN_ENDED:
            if "code" in p or "function" in p or "implement" in p:
                return Route([], "coder", "code generation → coder agent")
            return Route([], "researcher", "open question → researcher agent")

        skills: list[str] = []
        if "summarize" in p or "summary" in p:
            skills.append("summarize")
        if "extract" in p or "json" in p or "schema" in p:
            skills.append("extract_json")

        agent = "coder" if ("code" in p or "implement" in p) else "researcher"
        return Route(skills or ["summarize"], agent, "mixed → skills then agent")
