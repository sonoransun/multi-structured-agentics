"""Stub-mode tests for ToolUserAgent.

Stub backends don't emit `tool_calls`, so the agent's keyword-fallback
path is what these tests pin down. They lock in the shape of the result
and confirm the trace records at least one span (the LLM call).
"""
from __future__ import annotations

from msa.agents.tool_user import ToolUserAgent
from msa.core.llm import LLM
from msa.core.trace import Trace
from msa.skills.summarize import SummarizeSkill


def test_stub_dispatches_via_keyword_fallback() -> None:
    llm = LLM()  # stub mode — no API key
    assert llm.stub_mode

    agent = ToolUserAgent(llm, skills=[SummarizeSkill(llm)])
    trace = Trace()
    result = agent.handle(
        "summarize this passage: the quick brown fox jumps over the lazy dog",
        trace,
    )

    assert result.ok
    assert result.output is not None
    # Keyword fallback fires: result is a dict with both answer and summary.
    assert isinstance(result.output, dict)
    assert "summary" in result.output
    # At least the agent LLM span; the summarize skill adds another.
    assert len(trace.spans) >= 1


def test_no_skills_returns_text() -> None:
    llm = LLM()
    assert llm.stub_mode

    agent = ToolUserAgent(llm, skills=[])
    trace = Trace()
    result = agent.handle("hello there", trace)

    assert result.ok
    assert isinstance(result.output, str)
    assert result.output  # non-empty stub text
    assert len(trace.spans) >= 1
