"""Online bandit update for the LearnedRouter.

Reads collected JSONL, computes per-(kind, mode-label) reward averages,
and writes them as multiplicative biases to `models/bandit.json`. The
LearnedRouter applies these at inference time:

    proba[label] *= 1.0 + bandit[kind][label]

Negative biases discourage labels that consistently underperform; positive
ones reinforce winners. The bias is bounded to [-0.5, +0.5] so we don't
flip predictions outright — the classifier still drives, the bandit nudges.

Usage:
    python -m msa.training.bandit --data data/runs --out models/bandit.json

This is the simplest possible online policy: epsilon-free, per-cluster
mean reward. Swap to Thompson sampling or contextual bandits when the
data justifies the complexity.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .train_router import _mode_to_label


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/runs")
    p.add_argument("--out", default="models/bandit.json")
    p.add_argument("--clip", type=float, default=0.5, help="bound on |bias|")
    args = p.parse_args(argv)

    rows: list[dict] = []
    for jl in Path(args.data).glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print(f"no data in {args.data}", file=sys.stderr)
        return 1

    # (kind, label) -> [scores]
    bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    # Optional preference bias accumulator: rows produced by a DPO/reranker
    # pipeline may carry a `pref_bias` float per (kind, label). Sum it in.
    pref: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        label = _mode_to_label(r["mode"], r["kind"])
        bucket[(r["kind"], label)].append(r["score"])
        if "pref_bias" in r:
            pref[(r["kind"], label)].append(float(r["pref_bias"]))

    # Mean score per kind, used to center each label's bias relative to its peers.
    kind_means: dict[str, float] = {}
    for k in {kind for kind, _ in bucket}:
        all_scores = [s for (kk, _), ss in bucket.items() for s in ss if kk == k]
        kind_means[k] = sum(all_scores) / len(all_scores) if all_scores else 0.0

    bandit: dict[str, dict[str, float]] = defaultdict(dict)
    for (kind, label), scores in bucket.items():
        mean = sum(scores) / len(scores)
        bias = (mean - kind_means[kind])  # ∈ roughly [-1, +1]
        # Sum preference bias before clipping so it can pull labels up/down,
        # but stays bounded by --clip.
        if pref.get((kind, label)):
            bias += sum(pref[(kind, label)]) / len(pref[(kind, label)])
        bias = max(-args.clip, min(args.clip, bias))
        bandit[kind][label] = round(bias, 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bandit, indent=2))
    print(f"saved {out} ({sum(len(v) for v in bandit.values())} biases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
