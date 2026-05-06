"""Train a sklearn router classifier from collected JSONL.

Reads `data/runs/*.jsonl`, picks the best-scoring mode per task as the
label, fits logistic regression on the same hand-crafted features the
LearnedRouter uses at inference, and saves the bundle.

Usage:
    python -m msa.training.train_router --data data/runs --out models/router.joblib

If the data is sparse (each task only has one mode tried), the script
synthesizes a fallback label from `kind` so training still produces a
usable artifact — clearly worse than data-rich training, but better than
keyword-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..routing.learned import _features


def _best_mode_per_task(rows: list[dict]) -> dict[str, str]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    out = {}
    for tid, runs in by_task.items():
        winner = max(runs, key=lambda r: r["score"])
        out[tid] = winner["mode"]
    return out


def _mode_to_label(mode: str, kind: str) -> str:
    if mode == "skills_only":
        return "skill:extract_json" if kind == "structured" else "skill:summarize"
    if mode == "agents_only":
        return "agent:coder" if kind == "open_ended" else "agent:researcher"
    return "mixed:research"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/runs", help="dir containing JSONL run logs")
    p.add_argument("--out", default="models/router.joblib")
    args = p.parse_args(argv)

    try:
        from sklearn.linear_model import LogisticRegression
        import joblib
    except ImportError as e:
        print(f"missing dep: {e}. install with `pip install -e .[training]`", file=sys.stderr)
        return 2

    data_dir = Path(args.data)
    rows: list[dict] = []
    for jl in data_dir.glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print(f"no data in {data_dir}", file=sys.stderr)
        return 1

    best = _best_mode_per_task(rows)
    seen_tasks: set[str] = set()
    X: list[list[float]] = []
    y: list[str] = []
    for r in rows:
        if r["task_id"] in seen_tasks:
            continue
        seen_tasks.add(r["task_id"])
        X.append(_features(r["prompt"], r["kind"]))
        y.append(_mode_to_label(best[r["task_id"]], r["kind"]))

    if len(set(y)) < 2:
        print(f"only one label class observed ({y[0] if y else 'none'}); add diversity", file=sys.stderr)
        return 1

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": list(model.classes_)}, out_path)
    print(f"saved {out_path} (n={len(X)}, labels={list(model.classes_)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
