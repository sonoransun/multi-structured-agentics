"""Distillation: fine-tune a small local model on (prompt, claude_output) pairs.

The setup:
- Run benchmarks with `MSA_BACKEND=claude` to populate JSONL with
  high-score Claude outputs.
- Run this script to LoRA-tune a small base model (default `flan-t5-small`)
  to mimic those outputs on the skill prompts.
- Load the result via TransformersBackend; route skills to the local model
  via MSA_BACKEND_SKILL=transformers.

End state: skills run on a cheap model that reproduces Claude's behavior
on the narrow distribution they handle, while open-ended agent work
continues to call Claude. That's the multi-model thesis paying off.

Usage:
    python -m msa.training.distill \\
        --data data/runs \\
        --base google/flan-t5-small \\
        --out models/distilled \\
        --epochs 3 --min-score 0.8

Requires `[training]` extras. The script is intentionally compact —
it's a working starting point, not a state-of-the-art trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_pairs(data_dir: Path, min_score: float) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for jl in data_dir.glob("*.jsonl"):
        for line in jl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("score", 0.0) < min_score:
                continue
            output = row.get("output")
            if isinstance(output, dict):
                # Pull a textual representative
                output = output.get("answer") or output.get("summary") or output.get("code") or json.dumps(output)
            if not output:
                continue
            pairs.append({"input": row["prompt"], "target": str(output)})
    return pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/runs")
    p.add_argument("--base", default="google/flan-t5-small")
    p.add_argument("--out", default="models/distilled")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.8)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    args = p.parse_args(argv)

    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        print(f"missing dep: {e}. install with `pip install -e .[training]`", file=sys.stderr)
        return 2

    pairs = _load_pairs(Path(args.data), args.min_score)
    if len(pairs) < 4:
        print(
            f"not enough high-quality data ({len(pairs)} pairs, need ≥4). "
            "run `python bench.py` against Claude first to populate data/runs/.",
            file=sys.stderr,
        )
        return 1

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=0.05,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    def encode(batch: dict[str, list[str]]) -> dict[str, Any]:
        x = tok(batch["input"], truncation=True, padding="max_length", max_length=256)
        y = tok(batch["target"], truncation=True, padding="max_length", max_length=256)
        x["labels"] = y["input_ids"]
        return x

    ds = Dataset.from_list(pairs).map(encode, batched=True, remove_columns=["input", "target"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        learning_rate=1e-3,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tok, model=model),
    )
    trainer.train()
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"saved LoRA adapter + tokenizer to {out}")
    print(f"load via: TransformersBackend(model='{out}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
