from __future__ import annotations

import hashlib
import json

from .base import Backend, Response


class StubBackend(Backend):
    """Deterministic, dep-free backend.

    Sniffs the prompt for hints about expected output shape so downstream
    code can parse what it gets. This isn't a model — it's a fixture
    generator that keeps the harness honest about input/output contracts
    when no real backend is available.

    Always-available; the bottom of the fallback ladder.
    """

    name = "stub"
    available = True

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> Response:
        h = hashlib.sha256((system + "\n---\n" + prompt).encode()).hexdigest()[:12]
        lower = prompt.lower()

        if "json" in lower or "schema" in lower:
            text = json.dumps({"stub": True, "hash": h, "echo": prompt[:80]})
        elif "summarize" in lower or "summary" in lower:
            text = f"[stub summary {h}] " + " ".join(prompt.split()[:20])
        elif "code" in lower or "function" in lower or "implement" in lower:
            text = f"# stub code {h}\ndef solution():\n    return {h!r}\n"
        else:
            text = f"[stub {h}] {prompt[:120]}"

        return Response(
            text=text,
            tokens_in=len((system + prompt).split()),
            tokens_out=len(text.split()),
            backend=self.name,
        )
