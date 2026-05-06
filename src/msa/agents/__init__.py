from .base import Agent, AgentResult
from .coder import CoderAgent
from .researcher import ResearcherAgent
from .tool_user import ToolUserAgent

REGISTRY: dict[str, type[Agent]] = {
    "coder": CoderAgent,
    "researcher": ResearcherAgent,
    "tool_user": ToolUserAgent,
}

__all__ = [
    "Agent",
    "AgentResult",
    "CoderAgent",
    "REGISTRY",
    "ResearcherAgent",
    "ToolUserAgent",
]
