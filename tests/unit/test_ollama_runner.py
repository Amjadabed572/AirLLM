"""Unit tests for the Ollama/GGUF runner (the `ollama` module is mocked)."""
from __future__ import annotations

import sys
import types

import pytest

from airllm_bench.services import ollama_runner
from airllm_bench.services.ollama_runner import run_ollama


def _fake_ollama(generate):
    mod = types.ModuleType("ollama")
    mod.pull = lambda *a, **k: None
    mod.generate = generate
    return mod


def test_bad_quant_raises(sample_prompt) -> None:
    with pytest.raises(ValueError, match="quant must be one of"):
        run_ollama("nope", sample_prompt)


def test_import_error_is_recorded(monkeypatch, sample_prompt) -> None:
    monkeypatch.setitem(sys.modules, "ollama", None)  # makes `import ollama` raise
    m = run_ollama("q4", sample_prompt)
    assert m.failed
    assert "ImportError" in m.failure_reason


def test_success_populates_metrics(monkeypatch, sample_prompt) -> None:
    resp = {"prompt_eval_count": 10, "eval_count": 5, "load_duration": 1e9,
            "prompt_eval_duration": 2e9, "eval_duration": 1e9}
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama(lambda *a, **k: resp))
    m = run_ollama("q4", sample_prompt)
    assert not m.failed
    assert m.prompt_tokens == 10
    assert m.output_tokens == 5
    assert m.ttft_s == pytest.approx(3.0)         # load 1s + prefill 2s
    assert m.tpot_s == pytest.approx(0.25)        # 1s decode / (5-1)
    assert m.throughput_tok_s == pytest.approx(5.0)


def test_runtime_failure_is_recorded(monkeypatch, sample_prompt) -> None:
    def boom(*a, **k):
        raise RuntimeError("server died")

    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama(boom))
    m = run_ollama("q8", sample_prompt)
    assert m.failed
    assert "server died" in m.failure_reason


def test_rss_sampler_returns_float() -> None:
    assert ollama_runner._ollama_rss_gb() >= 0.0
