"""Training infrastructure tests.

Collector tests run unconditionally — that's the data flywheel and must
work everywhere. The trainer scripts are only smoke-tested when their
optional deps are present; otherwise we skip with a clear message.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.methods import run_task_suite
from benchmarks.tasks import TASKS
from msa.core.llm import LLM
from msa.core.trace import Trace
from msa.training import TraceCollector


def test_collector_writes_jsonl(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    with TraceCollector(path=path) as c:
        c.record(
            task_id="t",
            kind="structured",
            prompt="p",
            mode="integrated",
            output={"answer": "x"},
            score=0.5,
            trace=Trace(),
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["score"] == 0.5
    assert rows[0]["mode"] == "integrated"


def test_collector_handles_non_serializable_output(tmp_path: Path):
    class Weird:
        def __init__(self):
            self.x = "y"

    path = tmp_path / "run.jsonl"
    with TraceCollector(path=path) as c:
        c.record(
            task_id="t",
            kind="open_ended",
            prompt="p",
            mode="agents_only",
            output=Weird(),
            score=0.0,
            trace=None,
        )
    row = json.loads(path.read_text().splitlines()[0])
    assert isinstance(row["output"], str)


def test_collector_integrates_with_task_suite(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    with TraceCollector(path=path) as c:
        run_task_suite(TASKS[:2], llm=LLM(), collector=c)
    lines = path.read_text().splitlines()
    # 2 tasks × 3 modes = 6 rows
    assert len(lines) == 6


@pytest.mark.skipif(
    pytest.importorskip("sklearn", reason="sklearn not installed") is None,
    reason="sklearn not installed",
)
def test_train_router_smoke(tmp_path: Path):
    # Generate a tiny dataset with two distinct labels so LogisticRegression
    # has ≥2 classes to fit.
    data_dir = tmp_path / "runs"
    data_dir.mkdir()
    rows = [
        {"task_id": "a", "kind": "structured", "prompt": "extract json", "mode": "skills_only", "score": 1.0},
        {"task_id": "b", "kind": "open_ended", "prompt": "implement code", "mode": "agents_only", "score": 1.0},
    ]
    (data_dir / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    out = tmp_path / "router.joblib"
    from msa.training import train_router

    rc = train_router.main(["--data", str(data_dir), "--out", str(out)])
    assert rc == 0
    assert out.exists()


def test_bandit_smoke(tmp_path: Path):
    data_dir = tmp_path / "runs"
    data_dir.mkdir()
    rows = [
        {"task_id": "a", "kind": "structured", "prompt": "extract", "mode": "skills_only", "score": 0.9},
        {"task_id": "b", "kind": "structured", "prompt": "extract", "mode": "agents_only", "score": 0.3},
    ]
    (data_dir / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    out = tmp_path / "bandit.json"
    from msa.training import bandit

    rc = bandit.main(["--data", str(data_dir), "--out", str(out)])
    assert rc == 0
    bias = json.loads(out.read_text())
    assert "structured" in bias
    # The skill label should have a positive bias (it scored higher)
    assert any(v > 0 for v in bias["structured"].values())
