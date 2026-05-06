"""HumanEval loader (deferred).

Stub kept in place so ``bench.py --tasks humaneval`` reports a clear
message instead of an ImportError. Wire this up by mirroring
``gsm8k.load``: pull the prompt + canonical test from HuggingFace
``openai_humaneval`` and emit Tasks with ``expected={"python_unit_test": ...}``
which ``_grader.py`` already supports.
"""
from __future__ import annotations

from msa.core.types import Task


def load(n: int = 20, split: str = "test") -> list[Task]:
    raise NotImplementedError("HumanEval deferred — see CLAUDE.md")
