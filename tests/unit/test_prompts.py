"""Unit tests for the prompt set."""
from __future__ import annotations

import pytest

from airllm_bench.services.prompts import PROMPTS, by_name


def test_prompts_have_unique_names() -> None:
    names = [p.name for p in PROMPTS]
    assert len(names) == len(set(names))


def test_by_name_returns_prompt() -> None:
    assert by_name("short").name == "short"


def test_by_name_unknown_raises() -> None:
    with pytest.raises(KeyError):
        by_name("does-not-exist")


def test_long_context_is_longest() -> None:
    lengths = {p.name: len(p.text) for p in PROMPTS}
    assert lengths["long_context"] > lengths["short"]


def test_prompts_are_frozen() -> None:
    with pytest.raises(AttributeError):
        by_name("short").max_new_tokens = 99
