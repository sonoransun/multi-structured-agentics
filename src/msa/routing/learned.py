from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..core.types import Task, TaskKind
from .base import Route, Router

DEFAULT_MODEL_PATH = "models/router.joblib"
DEFAULT_LABELS = [
    "skill:summarize",
    "skill:extract_json",
    "agent:researcher",
    "agent:coder",
    "mixed:research",
    "mixed:code",
]


def _features(prompt: str, kind: str) -> list[float]:
    """Hand-crafted features. Kept simple and interpretable."""
    p = prompt.lower()
    return [
        len(prompt),
        prompt.count(" "),
        float("extract" in p or "json" in p or "schema" in p),
        float("summarize" in p or "summary" in p),
        float("code" in p or "function" in p or "implement" in p),
        float("why" in p or "how" in p or "explain" in p),
        float(kind == "structured"),
        float(kind == "open_ended"),
        float(kind == "mixed"),
    ]


def _label_to_route(label: str) -> Route:
    if label == "skill:summarize":
        return Route(["summarize"], None, "learned → summarize")
    if label == "skill:extract_json":
        return Route(["extract_json"], None, "learned → extract_json")
    if label == "agent:researcher":
        return Route([], "researcher", "learned → researcher")
    if label == "agent:coder":
        return Route([], "coder", "learned → coder")
    if label == "mixed:research":
        return Route(["summarize", "extract_json"], "researcher", "learned → mixed-research")
    if label == "mixed:code":
        return Route(["summarize"], "coder", "learned → mixed-code")
    return Route([], "researcher", f"learned → fallback ({label})")


class LearnedRouter(Router):
    """Router backed by a sklearn classifier loaded from disk.

    Training is offline (see training/train_router.py). At runtime we
    load the model once, then route is `predict(features(task))`.

    `available` is True iff the model file exists AND sklearn imports.
    Without a trained model, callers fall back to KeywordRouter.

    Bandit mode: if the bandit weights file exists, predictions are
    re-scored online using collected (feature, label, reward) tuples.
    See training/bandit.py for the update step.
    """

    name = "learned"

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, bandit_path: str | None = None):
        self.model_path = Path(model_path)
        self.bandit_path = Path(bandit_path) if bandit_path else Path("models/bandit.json")
        self._model: Any = None
        self._labels: list[str] = DEFAULT_LABELS
        self._bandit: dict[str, dict[str, float]] = {}
        self.available = self.model_path.exists() and self._sklearn_available()
        if self.available:
            self._load()

    @staticmethod
    def _sklearn_available() -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        import joblib

        bundle = joblib.load(self.model_path)
        self._model = bundle["model"]
        self._labels = bundle.get("labels", DEFAULT_LABELS)
        if self.bandit_path.exists():
            try:
                self._bandit = json.loads(self.bandit_path.read_text())
            except json.JSONDecodeError:
                self._bandit = {}

    def route(self, task: Task) -> Route:
        if not self.available:
            from .keyword import KeywordRouter

            return KeywordRouter().route(task)

        feats = _features(task.prompt, task.kind.value)
        proba = self._model.predict_proba([feats])[0]

        # Bandit reweighting: if we've recorded outcomes for nearby tasks,
        # bias the probabilities. The kind-bucket key is a coarse cluster.
        bucket = task.kind.value
        if bucket in self._bandit:
            for i, lbl in enumerate(self._labels):
                proba[i] *= 1.0 + self._bandit[bucket].get(lbl, 0.0)

        best_idx = int(proba.argmax())
        return _label_to_route(self._labels[best_idx])
