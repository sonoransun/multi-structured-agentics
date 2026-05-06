"""Smoke test — runs every mode against every task in stub mode.

This is the bare minimum the harness owes you: end-to-end shape check with
no API key required. If the architecture wires together at all, these
assertions pass; if any pass-through is broken, they fail with a clear
message about which mode/task combination is at fault.
"""
from __future__ import annotations

import pytest

from benchmarks.methods import METHODS
from benchmarks.tasks import TASKS
from msa.core.llm import LLM
from msa.modes import MODES


@pytest.fixture
def llm() -> LLM:
    return LLM()  # stub mode (no key set in CI)


def test_stub_mode_active(llm: LLM) -> None:
    assert llm.stub_mode, "test must run without an API key — set ANTHROPIC_API_KEY=<empty> or unset"


@pytest.mark.parametrize("mode_name,mode_fn", list(MODES.items()))
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_mode_runs(mode_name, mode_fn, task, llm) -> None:
    output, trace = mode_fn(task, llm)
    assert isinstance(output, dict), f"{mode_name}/{task.id} returned {type(output)}"
    assert trace is not None
    assert len(trace.spans) >= 1, f"{mode_name}/{task.id} produced no spans"


@pytest.mark.parametrize("method_name,method_fn", list(METHODS.items()))
def test_method_runs(method_name, method_fn, llm) -> None:
    res = method_fn(TASKS, llm=llm)
    assert isinstance(res, dict)
    if method_name == "task_suite":
        assert "rows" in res and len(res["rows"]) == len(TASKS) * len(MODES)
    elif method_name == "synergy":
        assert "verdict" in res
        assert len(res["tasks"]) == len(TASKS)
    elif method_name == "capability_matrix":
        assert "matrix" in res
