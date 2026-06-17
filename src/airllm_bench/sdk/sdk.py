"""SDK layer -- the single entry point for ALL business logic.

Every consumer (CLI, notebooks, future GUI/REST) goes through this facade; none
import the service modules directly. All tunables are read from config via the
Config manager, never hardcoded.
"""
from __future__ import annotations

import json
import os

from airllm_bench.services import analyze as analyze_svc
from airllm_bench.services import hardware as hardware_svc
from airllm_bench.services import model_selector
from airllm_bench.services.airllm_runner import run_airllm
from airllm_bench.services.baseline_runner import run_baseline
from airllm_bench.services.economics import Scenario
from airllm_bench.services.metrics import RunMetrics
from airllm_bench.services.ollama_runner import run_ollama
from airllm_bench.services.prompts import by_name
from airllm_bench.shared.config import Config

_RESULTS = "results"


class AirLLMBenchSDK:
    """Facade exposing hardware, baseline, AirLLM, Ollama, and analysis flows."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.config.validate_versions()
        os.makedirs(_RESULTS, exist_ok=True)

    # -- configuration helpers --------------------------------------------
    def _power(self) -> float:
        return float(self.config.get("experiment.avg_power_w", 15.0))

    def resolve_model(self) -> str:
        """Config model id, or auto-select from detected hardware."""
        if not self.config.get("model.auto_select", False):
            return str(self.config.get("model.model_id"))
        hw = self.detect_hardware()
        return model_selector.recommend(hw.ram_total_gb, hw.disk_free_gb).repo_id

    # -- hardware ----------------------------------------------------------
    def detect_hardware(self) -> hardware_svc.HardwareInfo:
        """Probe the current machine."""
        return hardware_svc.detect()

    def save_hardware(self) -> hardware_svc.HardwareInfo:
        """Detect and persist results/hardware.json."""
        info = self.detect_hardware()
        with open(f"{_RESULTS}/hardware.json", "w", encoding="utf-8") as handle:
            json.dump(info.to_dict(), handle, indent=2)
        return info

    def recommend_model(self) -> str:
        """Explain the model recommendation for the detected hardware."""
        hw = self.detect_hardware()
        choice = model_selector.recommend(hw.ram_total_gb, hw.disk_free_gb)
        return model_selector.explain(choice, hw.ram_total_gb, hw.disk_free_gb)

    # -- experiments -------------------------------------------------------
    def run_baseline(self, prompt_name: str) -> RunMetrics:
        """Naive baseline; pre-writes a failure record so a hard OOM kill counts."""
        model = self.resolve_model()
        out = f"{_RESULTS}/baseline_{prompt_name}.json"
        provisional = RunMetrics(label="baseline", model=model, quantization="fp16",
                                 prompt=prompt_name)
        provisional.failed = True
        provisional.expected_failure = True  # not-loading IS the intended result
        provisional.failure_reason = (
            "Process killed by OS during naive in-RAM load (hard OOM) — model "
            "weights exceed physical RAM; the load did not complete."
        )
        provisional.save(out)
        metrics = run_baseline(model, by_name(prompt_name), avg_power_w=self._power())
        metrics.prompt = prompt_name
        metrics.expected_failure = True  # the baseline is expected to not fit
        metrics.save(out)
        return metrics

    def run_airllm(self, quant: str, prompt_name: str) -> RunMetrics:
        """AirLLM layer-streaming run at the given quant level."""
        shards = str(self.config.get("experiment.layer_shards_path", "layer_shards"))
        shards = os.environ.get("AIRLLM_SHARDS", shards)
        os.makedirs(shards, exist_ok=True)
        metrics = run_airllm(self.resolve_model(), by_name(prompt_name), quant, shards,
                             avg_power_w=self._power())
        metrics.prompt = prompt_name
        metrics.save(f"{_RESULTS}/airllm_{quant}_{prompt_name}.json")
        return metrics

    def run_ollama(self, quant: str, prompt_name: str) -> RunMetrics:
        """GGUF quantization run via Ollama at the given quant level."""
        metrics = run_ollama(quant, by_name(prompt_name), avg_power_w=self._power())
        metrics.prompt = prompt_name
        metrics.save(f"{_RESULTS}/ollama_{quant}_{prompt_name}.json")
        return metrics

    def run_input_length_study(self) -> list[RunMetrics]:
        """Parameter study: Ollama Q4 across short/medium/long_context to measure
        how TTFT (prefill) scales with input length."""
        results = []
        for name in self.config.get("study.prompt_names",
                                    ["short", "medium", "long_context"]):
            results.append(self.run_ollama(self.config.get("study.quant", "q4"), name))
        return results

    # -- analysis ----------------------------------------------------------
    def scenario(self) -> Scenario:
        """Build the economic Scenario from config."""
        return Scenario.from_config(self.config.section("economics"))

    def analyze(self) -> str:
        """Regenerate the summary table, figures, and economics summary."""
        return analyze_svc.analyze(self.scenario())
