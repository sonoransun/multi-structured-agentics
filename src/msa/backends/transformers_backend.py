from __future__ import annotations

import os
from typing import Any, Callable

from .base import Backend, Response

DEFAULT_MODEL = "google/flan-t5-small"


class TransformersBackend(Backend):
    """In-process HuggingFace transformers model.

    Heavier than Ollama (loads model into the calling process), but useful
    when:

      - you've trained or LoRA-fine-tuned a model and want to load it
        directly (see training/distill.py, training/lora.py)
      - the deployment target can't run a separate Ollama daemon
      - you want the model graph in-process for embedding extraction or
        custom decoding

    Lazy-imports `transformers` so the dep is only required when the
    backend is actually instantiated. Use `[training]` extras to install:
        pip install -e .[training]
    """

    name = "transformers"

    def __init__(self, model: str | None = None, device: str = "cpu"):
        self.model_name = model or os.environ.get("MSA_HF_MODEL", DEFAULT_MODEL)
        self.device = device
        self._pipeline: Any = None
        self.available = False
        try:
            import transformers  # noqa: F401

            self.available = True
        except ImportError:
            pass

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline

        # Use text2text for instruction-tuned encoder-decoders (T5/Flan),
        # text-generation for causal LMs. Heuristic on the name.
        task = (
            "text2text-generation" if any(k in self.model_name.lower() for k in ("t5", "flan", "bart"))
            else "text-generation"
        )
        self._pipeline = pipeline(task, model=self.model_name, device=self.device)

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 512,
        cache_system: bool = True,
        tools: list[dict] | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> Response:
        if not self.available:
            raise RuntimeError(
                "TransformersBackend unavailable — install with `pip install -e .[training]`"
            )
        self._ensure_loaded()

        full = f"{system}\n\n{prompt}".strip() if system else prompt
        out = self._pipeline(full, max_new_tokens=max_tokens, do_sample=False)
        text = (
            out[0].get("generated_text", "")
            if isinstance(out, list) and out
            else str(out)
        )
        # Causal LMs echo the prompt; trim it.
        if text.startswith(full):
            text = text[len(full):].lstrip()

        return Response(
            text=text,
            tokens_in=len(full.split()),
            tokens_out=len(text.split()),
            backend=f"{self.name}:{self.model_name}",
        )
