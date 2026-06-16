"""Plot generation. Produces the figures the README must embed.

Regenerates everything from results/*.json so figures always match raw data.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = "figures"


def _save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def bar_metric(runs: dict[str, float], ylabel: str, title: str, fname: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(runs.keys())
    vals = [runs[k] for k in labels]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    plt.xticks(rotation=20, ha="right")
    return _save(fig, fname)


def grouped_ttft_tpot(data: dict[str, dict[str, float]], fname: str) -> str:
    """data[label] = {'ttft_ms':..., 'tpot_ms':...}"""
    labels = list(data.keys())
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ttft = [data[k]["ttft_ms"] for k in labels]
    tpot = [data[k]["tpot_ms"] for k in labels]
    ax.bar(x - w / 2, ttft, w, label="TTFT (prefill)", color="#4C72B0")
    ax.bar(x + w / 2, tpot, w, label="TPOT (decode)", color="#DD8452")
    ax.set_ylabel("milliseconds")
    ax.set_title("Prefill (TTFT) vs Decode (TPOT) per configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    return _save(fig, fname)


def break_even(scn, max_volume: int, fname: str) -> str:
    volumes = np.linspace(1, max_volume, 400)
    curves = scn.curves(volumes)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, y in curves.items():
        ax.plot(volumes, y, label=label, linewidth=2)
    be = scn.break_even()
    if be is not None and be <= max_volume:
        ax.axvline(be, color="grey", linestyle="--", alpha=0.7)
        ax.annotate(f"break-even\n~{be:,.0f} req",
                    xy=(be, scn.curves(np.array([be]))["API (third-party)"][0]),
                    xytext=(be * 1.05, ax.get_ylim()[1] * 0.6), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="grey"))
    ax.set_xlabel("Number of requests (over the amortization period)")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title("On-Prem vs API vs Cloud GPU: cumulative cost & break-even")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, fname)


def roofline(intensity_pts: dict[str, float], peak_flops: float, peak_bw: float,
             fname: str) -> str:
    """Simple roofline: y = min(peak_flops, peak_bw * arithmetic_intensity)."""
    ai = np.logspace(-2, 3, 200)
    perf = np.minimum(peak_flops, peak_bw * ai)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(ai, perf, color="black", linewidth=2, label="roofline")
    ridge = peak_flops / peak_bw
    ax.axvline(ridge, color="grey", linestyle=":", alpha=0.6)
    for name, x in intensity_pts.items():
        y = min(peak_flops, peak_bw * x)
        ax.scatter([x], [y], s=60, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attainable performance (GFLOP/s)")
    ax.set_title("Roofline: where prefill vs decode are limited")
    ax.legend()
    return _save(fig, fname)
