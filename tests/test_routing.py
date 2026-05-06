"""Routing tests."""
from __future__ import annotations

import pytest

from benchmarks.tasks import TASKS
from msa.routing import KeywordRouter, EmbeddingRouter, LearnedRouter, get_default_router


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_keyword_router_returns_route_for_every_task(task):
    r = KeywordRouter().route(task)
    assert r is not None
    # Every task gets at least one of: skills, agent
    assert r.skills_first or r.agent


def test_default_router_is_keyword_when_nothing_else_set(monkeypatch):
    monkeypatch.delenv("MSA_ROUTER", raising=False)
    r = get_default_router()
    assert isinstance(r, KeywordRouter)


def test_embedding_router_falls_back_when_unavailable():
    r = EmbeddingRouter()
    if not r.available:
        # Falls back via package-level get_default_router won't pick it,
        # but if instantiated directly its .route still works (delegates)
        out = r.route(TASKS[0])
        assert out.rationale  # got something back


def test_learned_router_falls_back_without_model():
    r = LearnedRouter(model_path="nonexistent.joblib")
    assert not r.available
    out = r.route(TASKS[0])
    assert out is not None  # falls back to keyword
