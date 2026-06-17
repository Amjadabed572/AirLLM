"""Analysis orchestration (Tasks 5.4 / 5.5 / visualization).

Turns raw results/*.json into the comparison table + all figures, and computes
the economic break-even from the config-driven Scenario.
"""
from __future__ import annotations

import glob
import json
import os

from airllm_bench.constants import (
    ROOFLINE_INTENSITY,
    ROOFLINE_PEAK_BW_GBPS,
    ROOFLINE_PEAK_GFLOPS,
)
from airllm_bench.services import plots
from airllm_bench.services.economics import Scenario, summarize

_TABLE_HEAD = (
    "| Config | Prompt | Quant | TTFT (s) | TPOT (ms) | tok/s | Total (s) | "
    "Peak RAM (GB) | Peak VRAM (GB) | Energy (Wh) | Status |"
)
_TABLE_SEP = "|" + "---|" * 11


def load_results(results_dir: str = "results") -> list[dict]:
    """Load every per-run JSON (excluding hardware/summary files)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) in {"hardware.json", "summary_table.md"}:
            continue
        with open(path, encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def _short_reason(reason: str) -> str:
    """First clause of a failure reason, for the status cell."""
    return reason.split("—")[0].split(".")[0].strip()[:60]


def _key(r: dict) -> str:
    """Figure key: label plus prompt when a prompt is recorded."""
    prompt = r.get("prompt", "")
    return f"{r.get('label', '')} [{prompt}]" if prompt else r.get("label", "")


def markdown_table(rows: list[dict]) -> str:
    """Render the comparison table from raw run records."""
    lines = [_TABLE_HEAD, _TABLE_SEP]
    for r in rows:
        if not r.get("failed"):
            status = "ok"
        elif r.get("expected_failure"):
            status = "expected OOM ✓ (bottleneck confirmed)"
        else:
            status = f"FAILED — {_short_reason(r.get('failure_reason', ''))}"
        lines.append(
            f"| {r.get('label', '')} | {r.get('prompt', '') or '—'} | "
            f"{r.get('quantization', '')} | "
            f"{r.get('ttft_s', 0):.2f} | {r.get('tpot_s', 0) * 1000:.1f} | "
            f"{r.get('throughput_tok_s', 0):.2f} | {r.get('total_gen_s', 0):.1f} | "
            f"{r.get('peak_ram_gb', 0):.1f} | "
            f"{r.get('peak_vram_gb', 0):.1f} | {r.get('est_energy_wh', 0):.2f} | {status} |"
        )
    return "\n".join(lines)


def make_figures(rows: list[dict], scenario: Scenario) -> list[str]:
    """Regenerate every figure from runs (if any) plus the economic models."""
    paths = []
    ok = [r for r in rows if not r.get("failed")]
    if ok:
        thr = {_key(r): r["throughput_tok_s"] for r in ok}
        paths.append(plots.bar_metric(thr, "tokens / sec",
                                      "Throughput by configuration", "throughput.png"))
        ram = {_key(r): r["peak_ram_gb"] for r in ok}
        paths.append(plots.bar_metric(ram, "GB", "Peak RAM by configuration", "peak_ram.png"))
        tt = {_key(r): {"ttft_ms": r["ttft_s"] * 1000, "tpot_ms": r["tpot_s"] * 1000}
              for r in ok}
        paths.append(plots.grouped_ttft_tpot(tt, "ttft_vs_tpot.png"))

        # Parameter study: TTFT vs input length, if one engine ran >=2 prompts.
        sweep = _input_length_points(ok)
        if len(sweep) >= 2:
            paths.append(plots.ttft_vs_input(sweep, "ttft_vs_input.png"))

    be = scenario.break_even()
    max_vol = int((be or 100_000) * 2.5)
    paths.append(plots.break_even(scenario, max_vol, "break_even.png"))
    paths.append(plots.roofline(ROOFLINE_INTENSITY, peak_flops=ROOFLINE_PEAK_GFLOPS,
                                peak_bw=ROOFLINE_PEAK_BW_GBPS, fname="roofline.png"))
    return paths


def _input_length_points(ok: list[dict]) -> list[tuple[int, float]]:
    """(prompt_tokens, ttft_s) for the engine+quant that ran the most prompts."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        groups[r.get("label", "")].append(r)
    best = max(groups.values(), key=len, default=[])
    pts = sorted((r.get("prompt_tokens", 0), r.get("ttft_s", 0.0)) for r in best)
    # Only meaningful if input length actually varies.
    return pts if len({p for p, _ in pts}) >= 2 else []


def analyze(scenario: Scenario, results_dir: str = "results") -> str:
    """Build the table + figures + economics summary; return the table markdown."""
    rows = load_results(results_dir)
    table = markdown_table(rows)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary_table.md"), "w", encoding="utf-8") as handle:
        handle.write(table + "\n")
    make_figures(rows, scenario)
    return table + "\n\n" + summarize(scenario)
