"""Multi-model policy tests."""
from __future__ import annotations

import pytest

from msa.backends import StubBackend
from msa.core.trace import Trace
from msa.orchestrator import Orchestrator
from msa.policy import MultiModelPolicy, Role
from benchmarks.tasks import TASKS


def test_policy_resolves_to_stub_in_clean_env(monkeypatch):
    for k in ("MSA_BACKEND", "MSA_BACKEND_SKILL", "MSA_BACKEND_AGENT", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    p = MultiModelPolicy()
    desc = p.describe()
    assert desc[Role.SKILL.value] == "stub"
    assert desc[Role.AGENT.value] == "stub"


def test_policy_accepts_explicit_backends():
    p = MultiModelPolicy(skill_backend=StubBackend(), agent_backend=StubBackend())
    assert p.llm_for(Role.SKILL).backend.name == "stub"
    assert p.llm_for(Role.AGENT).backend.name == "stub"


def test_orchestrator_with_policy_runs_end_to_end(monkeypatch):
    for k in ("MSA_BACKEND", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    p = MultiModelPolicy()
    orch = Orchestrator(policy=p)
    trace = Trace()
    out = orch.run(TASKS[0], trace)
    assert "policy" in out
    assert out["policy"][Role.SKILL.value]
