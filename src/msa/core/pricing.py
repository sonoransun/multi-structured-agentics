"""Backend cost pricing — USD per million tokens.

Stub/ollama/transformers all return 0.0 (no monetary cost). Claude prices
are looked up by model name. The list intentionally stays small and is the
source of truth for both the LLM facade (which attaches `cost_usd` to every
response) and the Backend Pareto benchmark method.
"""
from __future__ import annotations

# (input $/M, output $/M, cache_read $/M).
# Cache reads on Claude are roughly 10% of input cost.
PRICES_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "claude-opus-4-7": (15.0, 75.0, 1.5),
    "claude-opus-4-6": (15.0, 75.0, 1.5),
    "claude-sonnet-4-6": (3.0, 15.0, 0.3),
    "claude-haiku-4-5": (1.0, 5.0, 0.1),
}


def estimate_cost(
    *,
    backend: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cache_read_tokens: int = 0,
) -> float:
    if not backend or not backend.startswith("claude"):
        return 0.0
    key = model
    if key not in PRICES_PER_MTOK:
        for k in PRICES_PER_MTOK:
            if model.startswith(k):
                key = k
                break
        else:
            return 0.0
    in_p, out_p, cache_p = PRICES_PER_MTOK[key]
    fresh_in = max(0, tokens_in - cache_read_tokens)
    return (
        fresh_in * in_p
        + tokens_out * out_p
        + cache_read_tokens * cache_p
    ) / 1_000_000
