"""Tests for public benchmark loaders."""
from __future__ import annotations


def test_gsm8k_loads_bundled_sample_without_datasets():
    """Even if `datasets` is not installed, the bundled sample must work."""
    from benchmarks.datasets.gsm8k import load

    tasks = load(n=5)
    assert len(tasks) >= 1
    assert all(t.expected and "numeric_answer" in t.expected for t in tasks)


def test_gsm8k_grader_path():
    from benchmarks.methods._grader import score_output

    expected = {"numeric_answer": "42"}
    assert score_output("the answer is 42", expected) == 1.0
    assert score_output("the answer is 7", expected) == 0.0
    assert score_output("no number here", expected) == 0.0


def test_gsm8k_task_shape():
    """Tasks should have id prefix, OPEN_ENDED kind, and a usable prompt."""
    from benchmarks.datasets.gsm8k import load
    from msa.core.types import TaskKind

    tasks = load(n=3)
    assert len(tasks) >= 1
    for t in tasks:
        assert t.id.startswith("gsm8k_")
        assert t.kind == TaskKind.OPEN_ENDED
        assert "Provide the final numeric answer" in t.prompt


def test_humaneval_stub_raises():
    from benchmarks.datasets.humaneval import load

    try:
        load(n=1)
    except NotImplementedError as e:
        assert "HumanEval" in str(e)
    else:
        raise AssertionError("humaneval.load should raise NotImplementedError")
