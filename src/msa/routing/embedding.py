from __future__ import annotations

from typing import Any

from ..core.types import Task, TaskKind
from .base import Route, Router

DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Anchor texts used to score routes via cosine similarity. Each anchor
# represents a recipe; the router picks the highest-similarity one.
ANCHORS: dict[str, dict[str, Any]] = {
    "skill_extract": {
        "text": "extract a structured json object with named fields from text",
        "route": Route(["extract_json"], None, "embedding → extract_json"),
    },
    "skill_summarize": {
        "text": "produce a short one-sentence summary of a passage",
        "route": Route(["summarize"], None, "embedding → summarize"),
    },
    "agent_research": {
        "text": "answer an open-ended factual or analytical question",
        "route": Route([], "researcher", "embedding → researcher"),
    },
    "agent_code": {
        "text": "implement a python function that solves a problem",
        "route": Route([], "coder", "embedding → coder"),
    },
    "mixed_extract_research": {
        "text": "explain an issue and extract structured fields from a customer note",
        "route": Route(["summarize", "extract_json"], "researcher", "embedding → mixed"),
    },
}


class EmbeddingRouter(Router):
    """Semantic router using sentence-transformers.

    Compares the task prompt embedding to a small bank of anchor recipes
    and picks the highest-similarity route. Lazy-imports
    `sentence_transformers`; if missing, the router reports
    `available=False` and the package falls back to KeywordRouter.

    Why anchors over a learned classifier: the anchors are interpretable
    and editable without retraining. When the system grows new skills, you
    add an anchor — no data, no fitting. The LearnedRouter handles the
    case where the anchors aren't expressive enough.
    """

    name = "embedding"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: Any = None
        self._anchor_vecs: Any = None
        self.available = False
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401

            self.available = True
        except ImportError:
            return

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._anchor_vecs = self._model.encode(
            [a["text"] for a in ANCHORS.values()],
            normalize_embeddings=True,
        )

    def route(self, task: Task) -> Route:
        if not self.available:
            # Caller should have used the fallback, but be defensive
            from .keyword import KeywordRouter

            return KeywordRouter().route(task)
        self._ensure_loaded()

        import numpy as np

        q = self._model.encode([task.prompt], normalize_embeddings=True)[0]
        sims = np.asarray(self._anchor_vecs) @ q
        best_idx = int(np.argmax(sims))
        best_key = list(ANCHORS.keys())[best_idx]
        chosen = ANCHORS[best_key]["route"]
        # Respect explicit MIXED kind even if the anchor disagrees
        if task.kind == TaskKind.MIXED and chosen.agent is None:
            chosen = Route(
                chosen.skills_first or ["summarize"],
                "researcher",
                f"embedding+kind override → {best_key}",
            )
        return Route(
            chosen.skills_first,
            chosen.agent,
            f"{chosen.rationale} (sim={sims[best_idx]:.2f})",
        )
