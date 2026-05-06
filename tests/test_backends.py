"""Backend abstraction tests.

Stub backend always works; others are skipped when their preconditions
aren't met. Together they verify the facade does the right thing in every
deployment scenario.
"""
from __future__ import annotations

import pytest

from msa.backends import (
    ClaudeBackend,
    OllamaBackend,
    StubBackend,
    TransformersBackend,
)
from msa.core.llm import LLM


def test_stub_always_available():
    b = StubBackend()
    assert b.available
    r = b.complete("summarize this", system="be brief")
    assert r.text
    assert r.backend == "stub"


def test_stub_is_deterministic():
    b = StubBackend()
    a1 = b.complete("extract json from text").text
    a2 = b.complete("extract json from text").text
    assert a1 == a2


def test_facade_picks_stub_when_no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MSA_BACKEND", raising=False)
    llm = LLM()
    assert llm.stub_mode
    assert llm.backend.name == "stub"


def test_facade_respects_explicit_stub_env(monkeypatch):
    monkeypatch.setenv("MSA_BACKEND", "stub")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = LLM()
    assert llm.backend.name == "stub"


def test_facade_response_shape(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = LLM()
    r = llm.complete("test prompt")
    assert isinstance(r.text, str)
    assert isinstance(r.tokens_in, int)
    assert isinstance(r.tokens_out, int)


def test_claude_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    b = ClaudeBackend()
    assert not b.available
    with pytest.raises(RuntimeError):
        b.complete("hi")


def test_ollama_unavailable_when_no_daemon():
    # Default localhost test — assumes no ollama running in CI.
    b = OllamaBackend(host="http://127.0.0.1:1")  # impossible port
    assert not b.available


def test_transformers_reports_availability_correctly():
    b = TransformersBackend()
    # Either transformers is installed (rare in lean test env) or not;
    # both are valid. The contract is that .available reflects reality.
    if b.available:
        # Don't actually load a model — that's slow. Just check shape.
        assert b.name == "transformers"
    else:
        with pytest.raises(RuntimeError):
            b.complete("hi")


def test_stub_stream_chunks_deterministic():
    from msa.backends import StubBackend
    chunks = []
    resp = StubBackend().complete(
        "summarize: hello world this is a test",
        stream_callback=chunks.append,
    )
    assert len(chunks) >= 2, f"expected ≥2 chunks, got {len(chunks)}: {chunks}"
    assert "".join(chunks) == resp.text, f"chunks didn't reassemble: {chunks!r} vs {resp.text!r}"


def test_llm_facade_forwards_stream_callback():
    from msa.core.llm import LLM
    from msa.backends import StubBackend
    llm = LLM(backend=StubBackend())
    chunks = []
    resp = llm.complete("summarize: test", stream_callback=chunks.append)
    assert chunks
    assert "".join(chunks) == resp.text


def test_stub_no_callback_path_unchanged():
    """Regression: ensure default (no callback) still works."""
    from msa.backends import StubBackend
    resp = StubBackend().complete("summarize: hello")
    assert resp.text and resp.tokens_out > 0
