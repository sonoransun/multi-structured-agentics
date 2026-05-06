from __future__ import annotations

import json
import re
from typing import Any

from ..core.trace import Trace
from .base import Skill, SkillResult

SYSTEM = (
    "Extract a JSON object matching the requested schema from the user's text. "
    "Reply with ONLY the JSON object — no fences, no prose. If a field is not "
    "present in the text, use null."
)


class ExtractJSONSkill(Skill):
    """Extract structured fields from free text into a JSON object.

    The schema is supplied per-call via `inputs["schema"]` (a JSON-schema dict).
    This is the canonical "structured" operation: input is fuzzy, output is
    validated against a known shape.
    """

    name = "extract_json"
    description = "Extract a JSON object matching a given schema from free text."

    def run(self, inputs: dict[str, Any], trace: Trace) -> SkillResult:
        text = inputs.get("text", "")
        schema = inputs.get("schema", {})
        if not text or not schema:
            return SkillResult(output=None, ok=False, error="missing 'text' or 'schema'")

        prompt = (
            f"Schema:\n{json.dumps(schema, indent=2)}\n\nText:\n{text}\n\nReturn JSON only."
        )

        span = trace.open(self.name, "skill")
        try:
            resp = self.llm.complete(prompt, system=SYSTEM, max_tokens=1024)
            data = _parse_json(resp.text)
            ok = isinstance(data, dict)
            trace.close(span, resp.tokens_in, resp.tokens_out, note="extract_json", cost_usd=resp.cost_usd)
            return SkillResult(output=data, ok=ok, error=None if ok else "non-object output")
        except Exception as e:
            trace.close(span, note=f"error: {e}")
            return SkillResult(output=None, ok=False, error=str(e))


def _parse_json(text: str) -> Any:
    """Parse JSON, tolerating fences and trailing prose the model sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    # Find the first {...} balanced run; cheaper than a full grammar.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
