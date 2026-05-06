from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    kind: str  # "skill" | "agent" | "router"
    started_at: float
    ended_at: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    note: str = ""

    @property
    def duration_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at


@dataclass
class Trace:
    """Records the path a task takes through the system.

    The trace is the primary diagnostic output of a run — benchmark methods
    consume it to attribute cost, latency, and decisions to skills vs. agents.
    """

    spans: list[Span] = field(default_factory=list)

    def open(self, name: str, kind: str) -> Span:
        s = Span(name=name, kind=kind, started_at=time.time())
        self.spans.append(s)
        return s

    def close(self, span: Span, tokens_in: int = 0, tokens_out: int = 0, note: str = "") -> None:
        span.ended_at = time.time()
        span.tokens_in = tokens_in
        span.tokens_out = tokens_out
        span.note = note

    @property
    def tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.spans)

    @property
    def tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.spans)
