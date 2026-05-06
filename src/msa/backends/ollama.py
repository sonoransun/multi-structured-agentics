from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import Backend, Response

DEFAULT_MODEL = "llama3.2"
DEFAULT_HOST = "http://localhost:11434"


class OllamaBackend(Backend):
    """Local LLM via Ollama's HTTP API.

    Ollama is the simplest path to a local model — it ships a daemon that
    serves any quantized model on `localhost:11434`. We use stdlib urllib
    rather than `requests`/`httpx` to keep the dep footprint zero; the
    backend is "free" to depend on as long as Ollama is running.

    Configuration:
      MSA_OLLAMA_HOST  (default http://localhost:11434)
      MSA_OLLAMA_MODEL (default llama3.2)

    `available` is True iff a healthcheck succeeds at construction time.
    Caller should re-instantiate if the daemon comes up later.
    """

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.host = host or os.environ.get("MSA_OLLAMA_HOST", DEFAULT_HOST)
        self.model = model or os.environ.get("MSA_OLLAMA_MODEL", DEFAULT_MODEL)
        self.available = self._healthcheck()

    def _healthcheck(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=1.0) as r:
                return r.status == 200
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> Response:
        if not self.available:
            raise RuntimeError(f"OllamaBackend unavailable — no daemon at {self.host}")

        body = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama call failed: {e}") from e

        return Response(
            text=data.get("response", ""),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            backend=f"{self.name}:{self.model}",
        )
