from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.types import Task


@dataclass
class Route:
    skills_first: list[str]
    agent: str | None
    rationale: str
    backend_hint: str | None = None  # set by multi-model policy if it has an opinion


class Router(ABC):
    available: bool = True

    @abstractmethod
    def route(self, task: Task) -> Route: ...
