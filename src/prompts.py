"""Standard prompt set.

Using a fixed set keeps measurements comparable across baseline / AirLLM / each
quantization level. We include a short and a long prompt so you can show how
Prefill cost (TTFT) scales with input length vs Decode cost (TPOT).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prompt:
    name: str
    text: str
    max_new_tokens: int


PROMPTS: list[Prompt] = [
    Prompt(
        name="short",
        # max_new_tokens kept low: AirLLM streams ~15 GB from disk PER TOKEN on
        # this machine, so 20 tokens already gives a stable TPOT average while
        # keeping a single run to tens of minutes rather than hours. (The
        # assignment's Do-list explicitly recommends starting with few tokens.)
        text="In two sentences, explain what a transformer model is.",
        max_new_tokens=20,
    ),
    Prompt(
        name="medium",
        text=(
            "Explain the difference between the prefill and decode phases of "
            "LLM inference, and why one is compute-bound while the other is "
            "memory-bound. Use a concrete example."
        ),
        max_new_tokens=160,
    ),
    Prompt(
        name="long_context",
        # A deliberately long prompt to push prefill cost up (Task: show how
        # TTFT grows with input length). Repeat to inflate token count.
        text=(
            "You are reviewing a technical report. " + ("Context paragraph. " * 200)
            + "\n\nQuestion: summarize the key bottleneck described above."
        ),
        max_new_tokens=128,
    ),
]
