#!/usr/bin/env python3
"""CLI for the benchmark harness.

Usage:
    python bench.py                              # default: stub or auto-backend
    python bench.py --method synergy             # single method
    python bench.py --collect                    # write JSONL to data/runs/
    python bench.py --backend ollama             # force a single backend
    python bench.py --policy multi               # use MultiModelPolicy
    python bench.py --router embedding           # override router

Set ANTHROPIC_API_KEY for real Claude calls. Set MSA_OLLAMA_HOST (and run
`ollama serve`) to enable the local backend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from benchmarks.methods import METHODS
from benchmarks.tasks import TASKS
from msa.core.llm import LLM
from msa.training import TraceCollector


def main() -> int:
    p = argparse.ArgumentParser(description="msa benchmark runner")
    p.add_argument("--method", choices=list(METHODS) + ["all"], default="all")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    p.add_argument("--collect", action="store_true", help="write JSONL training data")
    p.add_argument("--data-dir", default=None, help="override data dir for collection")
    p.add_argument(
        "--backend",
        choices=["claude", "ollama", "transformers", "stub", "auto"],
        default="auto",
        help="single-backend mode (overrides --policy)",
    )
    p.add_argument(
        "--router",
        choices=["keyword", "embedding", "learned"],
        default=None,
        help="select router (also via MSA_ROUTER env)",
    )
    args = p.parse_args()

    if args.backend != "auto":
        os.environ["MSA_BACKEND"] = args.backend
    if args.router is not None:
        os.environ["MSA_ROUTER"] = args.router

    llm = LLM()
    runtime = (
        f"backend={llm.backend.name}  router={os.environ.get('MSA_ROUTER', 'keyword')}"
    )

    collector_ctx = (
        TraceCollector(path=args.data_dir + "/run.jsonl" if args.data_dir else None)
        if args.collect
        else None
    )

    methods = list(METHODS) if args.method == "all" else [args.method]
    results: dict[str, dict] = {}
    try:
        for name in methods:
            results[name] = METHODS[name](TASKS, llm=llm, collector=collector_ctx)
    finally:
        if collector_ctx is not None:
            collector_ctx.close()
            print(f"# collected → {collector_ctx.path}", file=sys.stderr)

    if args.json:
        print(json.dumps({"runtime": runtime, "results": results}, indent=2))
        return 0

    print(f"== msa benchmark — {runtime} ==\n")
    for name, res in results.items():
        print(f"--- {name} ---")
        _print_result(name, res)
        print()
    return 0


def _print_result(name: str, res: dict) -> None:
    if name == "task_suite":
        print(f"  mean score by mode: {res['summary']['mean_score']}")
        print(f"  mean tokens by mode: {res['summary']['mean_tokens']}")
        for r in res["rows"]:
            err = f"  ERR: {r['error']}" if r.get("error") else ""
            print(
                f"  {r['task']:14s} {r['mode']:14s} score={r['score']:.2f} "
                f"lat={r['latency_s']:.2f}s tok={r['tokens_in']}+{r['tokens_out']}{err}"
            )
    elif name == "synergy":
        print(f"  verdict: {res['verdict']}")
        print(f"  mean delta: {res['mean_delta']}  ({res['integration_wins']}/{res['n_tasks']} wins)")
        for d in res["tasks"]:
            print(
                f"  {d['task']:14s} kind={d['kind']:11s} "
                f"agent={d['agents_only']:.2f}  skill={d['skills_only']:.2f}  "
                f"integ={d['integrated']:.2f}  Δ={d['delta']:+.2f}"
            )
    elif name == "capability_matrix":
        print("  bin             agents_only  skills_only  integrated")
        for kind, scores in res["matrix"].items():
            a = scores.get("agents_only", 0.0)
            s = scores.get("skills_only", 0.0)
            i = scores.get("integrated", 0.0)
            print(f"  {kind:14s}  {a:>10.2f}   {s:>10.2f}   {i:>10.2f}")
        for note in res["interpretation"]:
            print(f"  - {note}")


if __name__ == "__main__":
    sys.exit(main())
