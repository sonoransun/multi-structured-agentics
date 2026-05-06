"""Preference-pair construction tests.

`build_pairs` is the dependency-free seam of the DPO pipeline — it must
work with stdlib only so the rest of the harness can be developed and
tested without `trl` / `transformers` installed.
"""
from __future__ import annotations

from msa.training.dpo import PrefPair, build_pairs


def test_build_pairs_basic():
    rows = [
        {"task_id": "t1", "prompt": "p", "output": "good", "score": 0.9},
        {"task_id": "t1", "prompt": "p", "output": "bad", "score": 0.2},
        {"task_id": "t2", "prompt": "q", "output": "x", "score": 0.5},
        {"task_id": "t2", "prompt": "q", "output": "y", "score": 0.55},  # gap < 0.3
    ]
    pairs = build_pairs(rows, min_gap=0.3)
    assert len(pairs) == 1
    assert pairs[0].chosen == "good"
    assert pairs[0].rejected == "bad"
    assert pairs[0].task_id == "t1"
    assert pairs[0].score_gap >= 0.3


def test_build_pairs_handles_dict_output():
    rows = [
        {"task_id": "t", "prompt": "p", "output": {"answer": "yes"}, "score": 0.9},
        {"task_id": "t", "prompt": "p", "output": {"answer": "no"}, "score": 0.1},
    ]
    pairs = build_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0].chosen == "yes"
    assert pairs[0].rejected == "no"


def test_build_pairs_skips_identical_text():
    rows = [
        {"task_id": "t", "prompt": "p", "output": "same", "score": 0.9},
        {"task_id": "t", "prompt": "p", "output": "same", "score": 0.1},
    ]
    assert build_pairs(rows) == []


def test_build_pairs_returns_dataclass():
    rows = [
        {"task_id": "x", "prompt": "p", "output": "a", "score": 1.0},
        {"task_id": "x", "prompt": "p", "output": "b", "score": 0.0},
    ]
    pairs = build_pairs(rows)
    assert all(isinstance(p, PrefPair) for p in pairs)


def test_build_pairs_no_data():
    assert build_pairs([]) == []
