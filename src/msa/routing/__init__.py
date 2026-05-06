"""Routing strategies.

The router decides, given a `Task`, which skills to run first and which
agent (if any) to invoke. Three implementations:

- KeywordRouter: hand-written rules. Free, transparent, brittle to wording.
- EmbeddingRouter: semantic similarity via sentence-transformers (optional).
  Falls back to KeywordRouter if the dep isn't installed.
- LearnedRouter: sklearn classifier trained on collected benchmark data.
  Improves with use; see training/train_router.py.

`route(task)` at the package level uses an env-selectable default so the
existing call site keeps working.
"""
from __future__ import annotations

import os

from ..core.types import Task
from .base import Route, Router
from .keyword import KeywordRouter
from .embedding import EmbeddingRouter
from .learned import LearnedRouter


def get_default_router() -> Router:
    name = os.environ.get("MSA_ROUTER", "keyword").lower()
    if name == "embedding":
        r = EmbeddingRouter()
        if r.available:
            return r
    if name == "learned":
        r = LearnedRouter()
        if r.available:
            return r
    return KeywordRouter()


_default: Router | None = None


def route(task: Task) -> Route:
    """Module-level entry point preserved for back-compat."""
    global _default
    if _default is None:
        _default = get_default_router()
    return _default.route(task)


__all__ = [
    "Route",
    "Router",
    "KeywordRouter",
    "EmbeddingRouter",
    "LearnedRouter",
    "route",
    "get_default_router",
]
