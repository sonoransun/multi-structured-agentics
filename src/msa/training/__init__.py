"""Custom training infrastructure.

Pipeline:

  1. collect.py     — JSONL writer wired into benchmark runs. Every
                      (task, mode, output, score, trace) becomes one line.
                      This is the data flywheel — without it, nothing else
                      can train.

  2. distill.py     — fine-tune a small local model on Claude's outputs.
                      Drop-in cheaper inference for the skill path.

  3. train_router.py — sklearn classifier from collected (features, best_mode)
                       pairs. Improves router accuracy over the keyword baseline.

  4. bandit.py      — online learner that biases the LearnedRouter based on
                      observed task scores. Updates after every benchmark run.

  5. lora.py        — LoRA / PEFT config for skill-specific fine-tuning.
                      Train one adapter per skill; load via TransformersBackend.

All modules gracefully report "missing optional deps" instead of crashing
at import time. The `[training]` extras install everything needed.
"""
from .collect import TraceCollector

__all__ = ["TraceCollector"]
