"""Standard prompt set (building block: fixed inputs).

A fixed prompt set keeps measurements comparable across baseline / AirLLM / each
quantization level. We include short, medium, and long prompts so prefill cost
(TTFT) can be studied as a function of input length vs. decode cost (TPOT).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """A single benchmark prompt.

    Input:  name (id), text (the prompt), max_new_tokens (decode budget).
    Output: immutable value object reused across every engine.
    """

    name: str
    text: str
    max_new_tokens: int


PROMPTS: list[Prompt] = [
    Prompt(
        name="short",
        # Low token budget: AirLLM streams ~15 GB from disk PER TOKEN on the 8 GB
        # target machine, so 20 tokens gives a stable TPOT average while keeping a
        # single run to tens of minutes (assignment Do-list: start with few tokens).
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
        # Deliberately long to push prefill cost up (show TTFT growth with input).
        text=(
            "You are reviewing a technical report. " + ("Context paragraph. " * 200)
            + "\n\nQuestion: summarize the key bottleneck described above."
        ),
        max_new_tokens=128,
    ),
]


def by_name(name: str) -> Prompt:
    """Return the prompt with the given name or raise KeyError."""
    for prompt in PROMPTS:
        if prompt.name == name:
            return prompt
    raise KeyError(f"unknown prompt: {name!r}")
