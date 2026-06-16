"""Unit tests for measurement primitives."""
from __future__ import annotations

import json

from airllm_bench.services.metrics import (
    MemorySampler,
    RunMetrics,
    time_streaming_generation,
)


def test_finalize_derives_tpot_and_throughput() -> None:
    m = RunMetrics(label="x", model="m", quantization="fp16")
    m.output_tokens, m.ttft_s, m.total_gen_s = 5, 1.0, 5.0
    m.finalize()
    assert m.tpot_s == (5.0 - 1.0) / (5 - 1)
    assert m.throughput_tok_s == 5 / 5.0


def test_finalize_handles_single_token() -> None:
    m = RunMetrics(label="x", model="m", quantization="q4")
    m.output_tokens, m.ttft_s, m.total_gen_s = 1, 0.5, 0.5
    m.finalize()
    assert m.tpot_s >= 0.0
    assert m.throughput_tok_s > 0.0


def test_save_writes_json(tmp_path, sample_metrics) -> None:
    out = tmp_path / "r.json"
    sample_metrics.save(str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["label"] == "airllm-fp16"
    assert data["output_tokens"] == 5


def test_memory_sampler_records_peak_and_energy() -> None:
    with MemorySampler(interval_s=0.01, avg_power_w=10.0) as sampler:
        _ = [0] * 100000
    assert sampler.peak_ram_gb > 0.0
    assert sampler.energy_wh >= 0.0


def test_time_streaming_generation_counts_tokens() -> None:
    m = RunMetrics(label="x", model="m", quantization="fp16")
    text = time_streaming_generation(iter(["a", "b", "c"]), m)
    assert text == "abc"
    assert m.output_tokens == 3
    assert m.ttft_s >= 0.0
    assert len(m.per_token_latencies_s) == 2
