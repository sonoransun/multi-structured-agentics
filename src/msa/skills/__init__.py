from .base import Skill, SkillResult
from .summarize import SummarizeSkill
from .extract_json import ExtractJSONSkill

REGISTRY: dict[str, type[Skill]] = {
    "summarize": SummarizeSkill,
    "extract_json": ExtractJSONSkill,
}

__all__ = ["Skill", "SkillResult", "SummarizeSkill", "ExtractJSONSkill", "REGISTRY"]
