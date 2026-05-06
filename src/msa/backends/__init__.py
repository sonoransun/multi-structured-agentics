"""LLM backend implementations.

Each backend implements `Backend.complete(prompt, system, max_tokens) -> Response`.
Selection happens in core/llm.py based on env vars or explicit config:

- ANTHROPIC_API_KEY set         → ClaudeBackend (default for production)
- MSA_BACKEND=ollama            → OllamaBackend (local, fast, free)
- MSA_BACKEND=transformers      → TransformersBackend (in-process HF model)
- otherwise                     → StubBackend (deterministic, dep-free)

Multi-model policy (policy/multi_model.py) can override per-call.
"""
from .base import Backend, Response
from .stub import StubBackend
from .claude import ClaudeBackend
from .ollama import OllamaBackend
from .transformers_backend import TransformersBackend

__all__ = [
    "Backend",
    "Response",
    "StubBackend",
    "ClaudeBackend",
    "OllamaBackend",
    "TransformersBackend",
]
