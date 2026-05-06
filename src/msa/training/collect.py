"""Trace collector.

Wired into the benchmark methods (see benchmarks/methods/*.py). Each row
captures a complete training example:

    {
      "task_id": "...",
      "kind": "structured",
      "prompt": "...",
      "mode": "integrated",
      "output": ...,
      "score": 0.83,
      "tokens_in": 200, "tokens_out": 87,
      "spans": [{"name": "router", "kind": "router", "duration_s": 0.0, ...}, ...],
    }

This is the source for every downstream learner: distillation reads
(prompt, output) where score==1, the router classifier reads (prompt,
mode) where mode is the task's argmax-score mode, and the bandit reads
(features, mode, score) directly.

Output goes to `data/runs/<run_id>.jsonl` by default. Set MSA_DATA_DIR
to redirect.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ..core.trace import Trace


class TraceCollector:
    def __init__(self, run_id: str | None = None, path: str | Path | None = None):
        if path is not None:
            self.path = Path(path)
        else:
            base = Path(os.environ.get("MSA_DATA_DIR", "data")) / "runs"
            base.mkdir(parents=True, exist_ok=True)
            run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
            self.path = base / f"{run_id}.jsonl"
        self._fh = self.path.open("a")

    def record(
        self,
        *,
        task_id: str,
        kind: str,
        prompt: str,
        mode: str,
        output: Any,
        score: float,
        trace: Trace | None,
        error: str | None = None,
        judge_score: float | None = None,
        cost_usd: float | None = None,
        backend: str | None = None,
        candidate_id: str | None = None,
        dataset: str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "task_id": task_id,
            "kind": kind,
            "prompt": prompt,
            "mode": mode,
            "output": _coerce(output),
            "score": score,
            "tokens_in": trace.tokens_in if trace else 0,
            "tokens_out": trace.tokens_out if trace else 0,
            "spans": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "duration_s": round(s.duration_s, 4),
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "note": s.note,
                    "cost_usd": s.cost_usd,
                    "judge_score": s.judge_score,
                }
                for s in (trace.spans if trace else [])
            ],
            "error": error,
        }
        if judge_score is not None:
            row["judge_score"] = judge_score
        if cost_usd is not None:
            row["cost_usd"] = cost_usd
        elif trace is not None:
            row["cost_usd"] = trace.cost_usd
        if backend is not None:
            row["backend"] = backend
        if candidate_id is not None:
            row["candidate_id"] = candidate_id
        if dataset is not None:
            row["dataset"] = dataset
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TraceCollector":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _coerce(x: Any) -> Any:
    """Make outputs JSON-serializable; fall back to repr for the long tail."""
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)
