"""Public benchmark loaders.

Each module exposes a ``load(n: int = 20, split: str = "test") -> list[Task]``
function. Heavy dependencies (e.g. HuggingFace ``datasets``) are imported
lazily inside ``load`` so the benchmark harness keeps working without them.
"""
