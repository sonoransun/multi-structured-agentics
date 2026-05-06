"""Benchmark task suite.

Tasks are deliberately span the structured/open-ended spectrum so the
capability matrix has signal in every cell. Each task carries:

- `kind`: how the router should classify it
- `expected`: a graded contract — judges check the actual output against it.
  Shape varies by task type; see `score_output` in methods/task_suite.py.
"""
from __future__ import annotations

from msa.core.types import Task, TaskKind

TASKS: list[Task] = [
    Task(
        id="t1_extract",
        prompt=(
            "Extract a JSON object with fields {name, role, company} from this bio: "
            "Jane Park is the CTO at Acme Robotics."
        ),
        kind=TaskKind.STRUCTURED,
        inputs={
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "company": {"type": "string"},
                },
                "required": ["name", "role", "company"],
            }
        },
        expected={"required_keys": ["name", "role", "company"]},
    ),
    Task(
        id="t2_summarize",
        prompt=(
            "Summarize: Photosynthesis is the process by which plants convert "
            "light energy into chemical energy stored in glucose, releasing oxygen "
            "as a byproduct. It happens primarily in the chloroplasts."
        ),
        kind=TaskKind.STRUCTURED,
        expected={"max_sentences": 2, "must_mention": ["plants"]},
    ),
    Task(
        id="t3_research",
        prompt="Why does adding salt to water raise its boiling point?",
        kind=TaskKind.OPEN_ENDED,
        expected={"must_mention": ["boiling"]},
    ),
    Task(
        id="t4_code",
        prompt="Implement a function `fizzbuzz(n)` that returns a list for 1..n.",
        kind=TaskKind.OPEN_ENDED,
        expected={"must_contain_code": ["def", "fizzbuzz"]},
    ),
    Task(
        id="t5_mixed",
        prompt=(
            "Read this customer note and explain the underlying issue, then extract "
            "a JSON {priority, category} for our ticket system. Note: 'My laptop "
            "screen is flickering badly since the last update — I have a demo "
            "tomorrow!'"
        ),
        kind=TaskKind.MIXED,
        inputs={
            "schema": {
                "type": "object",
                "properties": {
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "category": {"type": "string"},
                },
                "required": ["priority", "category"],
            }
        },
        expected={"must_mention": ["flicker"], "required_keys": ["priority", "category"]},
    ),
]
