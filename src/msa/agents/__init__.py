from .base import Agent, AgentResult
from .researcher import ResearcherAgent
from .coder import CoderAgent

REGISTRY: dict[str, type[Agent]] = {
    "researcher": ResearcherAgent,
    "coder": CoderAgent,
}

__all__ = ["Agent", "AgentResult", "ResearcherAgent", "CoderAgent", "REGISTRY"]
