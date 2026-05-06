"""Per-skill LoRA configuration and training entrypoint.

distill.py trains one adapter on all collected pairs. This script trains
one adapter *per skill* — `summarize.lora`, `extract_json.lora` — so a
deployment can swap in the right adapter for the role.

Usage:
    python -m msa.training.lora --skill summarize --data data/runs

Filters JSONL rows by `mode=skills_only` and the matching skill, then
calls into the same training loop as distill.py with skill-tagged output.

Why this matters: skills are narrower than full agent reasoning, so a
small adapter can saturate them without hurting the base model on
unrelated tasks. Run distill.py for general capability, run this for
skill-level specialization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import distill


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skill", required=True, help="skill name, e.g. summarize")
    p.add_argument("--data", default="data/runs")
    p.add_argument("--base", default="google/flan-t5-small")
    p.add_argument("--out-dir", default="models/skills")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.8)
    args = p.parse_args(argv)

    # Filter to rows that touched this skill at high score, write a
    # narrowed JSONL, then delegate.
    src = Path(args.data)
    out = Path(args.out_dir) / args.skill
    out.mkdir(parents=True, exist_ok=True)
    filtered = out / "_filtered.jsonl"
    n = 0
    with filtered.open("w") as fh:
        for jl in src.glob("*.jsonl"):
            for line in jl.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                spans = r.get("spans", [])
                touched = any(s.get("name") == args.skill and s.get("kind") == "skill" for s in spans)
                if touched and r.get("score", 0.0) >= args.min_score:
                    fh.write(line + "\n")
                    n += 1

    if n < 4:
        print(f"only {n} qualifying rows for skill={args.skill}; aborting", file=sys.stderr)
        return 1

    print(f"distilling skill={args.skill} on {n} examples → {out}")
    return distill.main(
        [
            "--data",
            str(filtered.parent),
            "--base",
            args.base,
            "--out",
            str(out),
            "--epochs",
            str(args.epochs),
            "--min-score",
            str(args.min_score),
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
