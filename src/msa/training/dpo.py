"""Preference / DPO training.

Two complementary uses of preference data:

  1. **Reranker** (`rerank` subcommand) — fit a cross-encoder that scores
     (prompt, candidate) pairs. Plug it into MultiModelPolicy as the
     `reranker` callable; when the orchestrator generates multiple
     candidates per step, the reranker picks the best.

  2. **DPO adapter** (`adapter` subcommand) — LoRA-DPO fine-tune of a
     small base model on (chosen, rejected) pairs via `trl.DPOTrainer`.
     Result loads via TransformersBackend.

Both consume preference pairs derived from collected JSONL: group rows
by task_id, emit (chosen=high-score, rejected=low-score) for every pair
whose score gap clears `--min-gap`.

Fail-soft: heavy deps (sentence-transformers, trl) are optional. If
unavailable, the script prints an install hint and returns 2. Importing
this module — including `build_pairs` — never requires them.

Usage:
    python -m msa.training.dpo rerank  --data data/runs --out models/preference
    python -m msa.training.dpo adapter --data data/runs --out models/dpo
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrefPair:
    task_id: str
    prompt: str
    chosen: str
    rejected: str
    score_gap: float


def _row_text(row: dict) -> str:
    out = row.get("output")
    if isinstance(out, dict):
        out = out.get("answer") or out.get("summary") or out.get("code") or json.dumps(out)
    return "" if out is None else str(out)


def build_pairs(rows: list[dict], min_gap: float = 0.3) -> list[PrefPair]:
    """Group rows by task_id; emit (chosen, rejected) for each pair with score_diff >= min_gap."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_id", "")].append(r)

    pairs: list[PrefPair] = []
    for tid, group in by_task.items():
        for i, hi in enumerate(group):
            for j, lo in enumerate(group):
                if i == j:
                    continue
                gap = float(hi.get("score", 0.0)) - float(lo.get("score", 0.0))
                if gap < min_gap:
                    continue
                ct, rt = _row_text(hi), _row_text(lo)
                if not ct or not rt or ct == rt:
                    continue
                pairs.append(
                    PrefPair(
                        task_id=tid,
                        prompt=hi.get("prompt", ""),
                        chosen=ct,
                        rejected=rt,
                        score_gap=round(gap, 4),
                    )
                )
    return pairs


def _load_rows(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for jl in data_dir.glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_pref_reranker(
    pairs: list[PrefPair],
    *,
    base: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    out: str = "models/preference",
) -> Path:
    """Fit a cross-encoder reranker on preference pairs. Returns output path."""
    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise RuntimeError(
            f"missing dep: {e}. install with `pip install -e .[embed,local]`"
        ) from e

    examples: list = []
    for p in pairs:
        examples.append(InputExample(texts=[p.prompt, p.chosen], label=1.0))
        examples.append(InputExample(texts=[p.prompt, p.rejected], label=0.0))

    model = CrossEncoder(base, num_labels=1)
    loader = DataLoader(examples, shuffle=True, batch_size=8)
    model.fit(train_dataloader=loader, epochs=1, warmup_steps=10, show_progress_bar=False)

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    return out_path


def train_dpo_adapter(
    pairs: list[PrefPair],
    *,
    base: str = "google/flan-t5-small",
    out: str = "models/dpo",
    lora_r: int = 8,
) -> Path:
    """LoRA-DPO fine-tune via trl.DPOTrainer. Returns output path."""
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise RuntimeError(
            f"missing dep: {e}. install with `pip install -e .[dpo]`"
        ) from e

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q", "v"],
        lora_dropout=0.05,
    )
    ds = Dataset.from_list(
        [{"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected} for p in pairs]
    )
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = DPOConfig(output_dir=str(out_path), num_train_epochs=1, per_device_train_batch_size=2, report_to=[])
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok, peft_config=lora_cfg)
    trainer.train()
    trainer.save_model(str(out_path))
    tok.save_pretrained(str(out_path))
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="msa-dpo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rerank", help="train cross-encoder reranker from prefs")
    pr.add_argument("--data", default="data/runs")
    pr.add_argument("--out", default="models/preference")
    pr.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    pr.add_argument("--min-gap", type=float, default=0.3)

    pa = sub.add_parser("adapter", help="LoRA-DPO via trl.DPOTrainer")
    pa.add_argument("--data", default="data/runs")
    pa.add_argument("--out", default="models/dpo")
    pa.add_argument("--base", default="google/flan-t5-small")
    pa.add_argument("--lora-r", type=int, default=8)
    pa.add_argument("--min-gap", type=float, default=0.3)

    args = p.parse_args(argv)

    rows = _load_rows(Path(args.data))
    if not rows:
        print(f"no data in {args.data}", file=sys.stderr)
        return 1
    pairs = build_pairs(rows, min_gap=args.min_gap)
    if len(pairs) < 2:
        print(f"only {len(pairs)} preference pair(s); need diverse-score data", file=sys.stderr)
        return 1

    try:
        if args.cmd == "rerank":
            path = train_pref_reranker(pairs, base=args.base, out=args.out)
        else:
            path = train_dpo_adapter(pairs, base=args.base, out=args.out, lora_r=args.lora_r)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"saved {path} (n_pairs={len(pairs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
