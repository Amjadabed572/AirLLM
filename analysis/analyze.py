"""Turn raw results/*.json into the comparison table + all figures.

    python -m analysis.analyze

Produces:
  results/summary_table.md   -> paste/embed into README & report
  figures/*.png              -> embedded in README & report
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import plots  # noqa: E402
from analysis.economics import Scenario  # noqa: E402


def load_results() -> list[dict]:
    rows = []
    for path in sorted(glob.glob("results/*.json")):
        if os.path.basename(path) in {"hardware.json", "summary_table.md"}:
            continue
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def markdown_table(rows: list[dict]) -> str:
    head = ("| Config | Prompt | Quant | TTFT (s) | TPOT (ms) | tok/s | "
            "Peak RAM (GB) | Peak VRAM (GB) | Energy (Wh) | Status |")
    sep = "|" + "---|" * 10
    lines = [head, sep]
    for r in rows:
        status = "FAILED" if r.get("failed") else "ok"
        prompt = r.get("label", "")
        lines.append(
            f"| {r.get('label','')} | {prompt} | {r.get('quantization','')} | "
            f"{r.get('ttft_s',0):.2f} | {r.get('tpot_s',0)*1000:.1f} | "
            f"{r.get('throughput_tok_s',0):.2f} | {r.get('peak_ram_gb',0):.1f} | "
            f"{r.get('peak_vram_gb',0):.1f} | {r.get('est_energy_wh',0):.2f} | {status} |"
        )
    return "\n".join(lines)


def make_figures(rows: list[dict]) -> list[str]:
    ok = [r for r in rows if not r.get("failed")]
    paths = []
    if ok:
        thr = {r["label"]: r["throughput_tok_s"] for r in ok}
        paths.append(plots.bar_metric(thr, "tokens / sec",
                                      "Throughput by configuration", "throughput.png"))
        ram = {r["label"]: r["peak_ram_gb"] for r in ok}
        paths.append(plots.bar_metric(ram, "GB", "Peak RAM by configuration",
                                      "peak_ram.png"))
        tt = {r["label"]: {"ttft_ms": r["ttft_s"] * 1000, "tpot_ms": r["tpot_s"] * 1000}
              for r in ok}
        paths.append(plots.grouped_ttft_tpot(tt, "ttft_vs_tpot.png"))

    scn = Scenario()
    be = scn.break_even()
    max_vol = int((be or 100_000) * 2.5)
    paths.append(plots.break_even(scn, max_vol, "break_even.png"))
    paths.append(plots.roofline({"prefill": 50.0, "decode": 0.3},
                                peak_flops=2000.0, peak_bw=50.0, fname="roofline.png"))
    return paths


def main() -> None:
    rows = load_results()
    if not rows:
        print("No results found in results/. Run the experiments first.")
        # Still emit economic + roofline figures so the pipeline is testable.
    table = markdown_table(rows)
    os.makedirs("results", exist_ok=True)
    with open("results/summary_table.md", "w") as f:
        f.write(table + "\n")
    figs = make_figures(rows)
    print(table)
    print("\nFigures written:")
    for p in figs:
        print(f"  {p}")
    from analysis.economics import summarize
    print("\nEconomics:\n" + summarize(Scenario()))


if __name__ == "__main__":
    main()
