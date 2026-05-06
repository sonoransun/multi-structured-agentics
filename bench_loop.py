"""Closed-loop self-improvement (Track T8).

Per iteration: collect synergy traces -> hash data -> train router ->
train bandit -> re-bench with `MSA_ROUTER=learned` -> record delta.

Steps 3 & 4 fail-soft when sklearn / training scripts aren't installed
(stub-mode CI is the canonical example). The collector and re-bench
always run, so the loop produces a meaningful report regardless.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IterationReport:
    iter: int
    runs_hash: str
    synergy_mean_delta: float
    synergy_delta_vs_prev: float
    router_skipped: bool
    bandit_skipped: bool


def _hash_runs(data_dir: Path) -> str:
    """sha1 over sorted (name, size, mtime) tuples for *.jsonl files."""
    if not data_dir.exists():
        return ""
    h = hashlib.sha1()
    for p in sorted(data_dir.glob("*.jsonl"), key=lambda x: x.name):
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _read_fp(path: Path) -> str:
    return path.read_text().strip() if path.exists() else ""


def _write_fp(path: Path, fp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fp)


def _train(kind: str, data_dir: Path, models_dir: Path) -> bool:
    """Run train_router or bandit; True = newly trained, False = skipped/failed.

    Skips silently when the input fingerprint hasn't changed since the
    previous successful run, or when the training module / its deps are
    missing (stub-mode CI without sklearn falls into this branch).
    """
    fp_path = models_dir / f".{kind}.fingerprint"
    cur_fp = _hash_runs(data_dir)
    if cur_fp and cur_fp == _read_fp(fp_path):
        return False
    try:
        if kind == "router":
            from msa.training import train_router as mod
            out = models_dir / "router.joblib"
        else:
            from msa.training import bandit as mod
            out = models_dir / "bandit.json"
    except ImportError:
        return False
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = mod.main(["--data", str(data_dir), "--out", str(out)])
    except Exception:
        return False
    if rc == 0:
        _write_fp(fp_path, cur_fp)
        return True
    return False


def _reset_router_cache() -> None:
    """msa.routing memoizes its default router; reset so a fresh
    MSA_ROUTER env value actually takes effect."""
    try:
        import msa.routing as r
        r._default = None
    except Exception:
        pass


def _rebench_synergy(tasks, models_dir: Path) -> float:
    """Re-run synergy with MSA_ROUTER=learned (if model exists), no collector."""
    from benchmarks.methods import run_synergy
    from msa.core.llm import LLM

    prev_router = os.environ.get("MSA_ROUTER")
    if (models_dir / "router.joblib").exists():
        os.environ["MSA_ROUTER"] = "learned"
    _reset_router_cache()
    try:
        result = run_synergy(tasks, llm=LLM(), collector=None)
    finally:
        if prev_router is None:
            os.environ.pop("MSA_ROUTER", None)
        else:
            os.environ["MSA_ROUTER"] = prev_router
        _reset_router_cache()
    return float(result.get("mean_delta", 0.0))


def run_self_improve(
    n: int,
    *,
    llm,
    tasks,
    data_dir: str = "data/runs",
    models_dir: str = "models",
) -> list[IterationReport]:
    from benchmarks.methods import run_synergy
    from msa.training import TraceCollector

    data_path = Path(data_dir)
    models_path = Path(models_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    reports: list[IterationReport] = []
    prev_hash = ""
    prev_delta = 0.0

    for i in range(1, n + 1):
        # 1. Collect a fresh synergy run into iter_{i}.jsonl.
        with TraceCollector(path=data_path / f"iter_{i}.jsonl") as c:
            run_synergy(tasks, llm=llm, collector=c)

        # 2. Hash data dir; break early if nothing changed.
        cur_hash = _hash_runs(data_path)
        if i > 1 and cur_hash == prev_hash:
            break

        # 3-4. Train router & bandit, both fail-soft.
        router_ok = _train("router", data_path, models_path)
        bandit_ok = _train("bandit", data_path, models_path)

        # 5. Re-bench with the learned router (no collector pollution).
        mean_delta = _rebench_synergy(tasks, models_path)

        # 6. Compute delta-vs-prev (0 on first iter by convention).
        delta_vs_prev = mean_delta - prev_delta if i > 1 else 0.0

        reports.append(
            IterationReport(
                iter=i,
                runs_hash=cur_hash,
                synergy_mean_delta=round(mean_delta, 4),
                synergy_delta_vs_prev=round(delta_vs_prev, 4),
                router_skipped=not router_ok,
                bandit_skipped=not bandit_ok,
            )
        )

        # 7. Convergence break (only meaningful from iter 2 onward).
        if i >= 2 and abs(delta_vs_prev) < 0.01:
            break

        prev_hash = cur_hash
        prev_delta = mean_delta

    return reports


def main() -> int:
    p = argparse.ArgumentParser(description="msa self-improvement loop")
    p.add_argument("-n", "--iters", type=int, default=2)
    p.add_argument("--data-dir", default="data/runs")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    from benchmarks.tasks import TASKS
    from msa.core.llm import LLM

    reports = run_self_improve(
        args.iters,
        llm=LLM(),
        tasks=list(TASKS),
        data_dir=args.data_dir,
        models_dir=args.models_dir,
    )
    if args.json:
        print(json.dumps([r.__dict__ for r in reports], indent=2, default=str))
    else:
        for r in reports:
            print(
                f"iter={r.iter} synergy_delta={r.synergy_mean_delta:+.3f} "
                f"vs_prev={r.synergy_delta_vs_prev:+.3f} "
                f"router_skipped={r.router_skipped} bandit_skipped={r.bandit_skipped}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
