# Deep-Dive Technical Report — EX05

**Running a Massive LLM Locally with AirLLM, Quantization & Performance Benchmarking**
Course L08 · Dr. Yoram Segal · June 2026

> This report documents a real, instrumented experiment. Example figures and
> numbers are committed so the document is complete; **replace the bracketed
> placeholders `[...]` and the example table with your measured values** before
> submission. Figures regenerate from `results/*.json` via `analysis/analyze.py`.

---

## Abstract

We attempt to run a [14B-parameter] model on-premises on hardware that cannot
hold it whole. A naive load fails (the bottleneck), so we route the model
through **AirLLM**, which streams one transformer layer at a time from disk, and
sweep **FP16 → Q8 → Q4** quantization. We measure TTFT, TPOT, throughput, peak
memory, and energy, then build a transparent **break-even** model comparing
local inference to a third-party API. The thesis we test with data — not
assumptions — is *when local deployment is economically rational and when an API
is the right call.*

---

## 1. Hardware & model selection (Task 5.1)

Auto-detected via `src/hardware.py` (`results/hardware.json`):

> Paste your `python -m src.hardware` output here.

| CPU | RAM | GPU / VRAM | Disk | Python |
|---|---|---|---|---|
| [Intel i7-12700H, 14c] | [16 GB] | [RTX 3060 / 6 GB or none] | [NVMe, 380 GB free] | [3.11] |

**Chosen model:** `[Qwen/Qwen2.5-14B-Instruct]`. The choice is deliberate: a
model whose FP16 footprint (~28 GB) **exceeds available RAM/VRAM**, guaranteeing
the naive baseline fails and making the AirLLM win measurable — yet whose shards
fit on disk and whose single layer fits in RAM, so the experiment is tractable
within the time budget. We use the **general `airllm.AutoModel`** class so the
correct architecture is matched automatically (avoiding the Qwen class-mismatch
error). Justification follows the assignment's "big enough to hurt, not
impossible" guidance.

---

## 2. Methodology (Tasks 5.2–5.4)

**Baseline (5.2).** `transformers.AutoModelForCausalLM.from_pretrained(...,
device_map="auto", torch_dtype=fp16)`. We expect — and on our hardware observed —
`[OOM / process kill / unusable latency]`. The failure is captured verbatim in
`results/baseline_*.json`; it is the bottleneck evidence, not an error to hide.

**AirLLM + quantization (5.3).** The same prompt set through `airllm.AutoModel`
with `layer_shards_saving_path` set to a **fast non-OS drive**, swept over
`compression ∈ {None(fp16), 8bit(Q8), 4bit(Q4)}`.

**Metrics (5.4).** Defined precisely:
- **TTFT** — start → first token (proxy for prefill cost / KV-cache build).
- **ITL / TPOT** — mean gap between subsequent tokens (decode cost).
- **Throughput** — output tokens / generation seconds.
- **Peak RAM / VRAM** — sampled on a background thread (catches the peak, not
  the end state).
- **Energy (Wh)** — assumed avg package power `[65 W]` × wall-time; document the
  wattage (use a wall meter / RAPL for precision).

All raw numbers are persisted so every figure is reproducible.

---

## 3. Results & analysis

Example results (replace with yours):

| Config | Quant | TTFT (s) | TPOT (ms) | tok/s | Peak RAM (GB) | Energy (Wh) | Status |
|---|---|---|---|---|---|---|---|
| baseline | fp16 | — | — | — | — | — | **FAILED (OOM)** |
| airllm | fp16 | 9.80 | 1850 | 0.53 | 7.9 | 38.0 | ok |
| airllm | q8 | 7.10 | 1200 | 0.81 | 6.1 | 24.0 | ok |
| airllm | q4 | 5.30 | 780 | 1.24 | 4.4 | 14.0 | ok |

![Throughput](../figures/throughput.png)
![TTFT vs TPOT](../figures/ttft_vs_tpot.png)
![Peak RAM](../figures/peak_ram.png)

**Interpretation.** Quantization to Q4 cut peak RAM by ~`[44%]` and TPOT by
~`[58%]` vs FP16, more than doubling throughput — because fewer bytes per weight
means less memory traffic in the bandwidth-bound decode phase. Output quality
degraded `[qualitatively: minor/noticeable]` at Q4 — this is the accuracy "red
line": past a point, smaller weights stop being free.

---

## 4. Research questions (Task 4)

1. **What was the bottleneck on the direct run — memory or compute?** Memory.
   The naive load `[OOM'd before generating]`; AirLLM made it run by trading RAM
   for disk I/O, confirming a **memory-capacity** bottleneck first, then a
   **memory-bandwidth + disk-I/O** bottleneck during decode.
2. **How does AirLLM change resource allocation?** It keeps only the active
   layer resident and pages the rest from disk — the virtual-memory analogy for
   model weights.
3. **Effect of quantization on memory, speed, paging, quality?** Smaller weights
   → less to page and stream → lower peak RAM and faster decode, until accuracy
   drops at Q4.
4. **How do Prefill/Decode show up as TTFT vs TPOT?** TTFT tracks prefill
   (compute-bound, grows with prompt length); TPOT tracks decode (memory-bound,
   roughly flat per token). See the long-context prompt for TTFT scaling.
5. **The Latency/Throughput price of running a big model on modest hardware?**
   `[seconds-scale TTFT, ~1 tok/s]` — usable for batch/offline, painful for
   interactive use.
6. **When is local economically worth it vs an external API?** See §6.

---

## 5. Inference-concept analysis (Task 5.6)

**Prefill is compute-bound; decode is memory-bound.** Prefill does one large
parallel matmul over all prompt tokens — lots of FLOPs, high arithmetic
intensity, sits near the flat (compute) ceiling of the roofline. Decode produces
one token at a time and must move the *entire* weight set through memory per
token — low arithmetic intensity, sits on the sloped (bandwidth) part of the
roofline. AirLLM's per-layer disk reads add a second, slower "memory" tier,
which is why decode dominates wall-time.

**Virtual memory / paging analogy.** AirLLM is to model layers what OS paging is
to RAM pages: resident working set kept tiny, the rest on disk, faulted in on
demand. The cost moves from "do you have enough RAM?" to "how fast is your
disk?" — `SSD/NVMe I/O` is the new ceiling.

![Roofline](../figures/roofline.png)

---

## 6. Economic analysis & recommendation (Task 5.5)

Two transparent models (`analysis/economics.py`), all assumptions stated:

- **API:** `requests × (in·price_in + out·price_out)`. We also model
  **prompt/context caching** (a discount on the repeated prefix), which can
  sharply lower API cost for repetitive long-context workloads and **push the
  break-even rightward**.
- **On-Prem:** amortized hardware CAPEX `[$2500 / 3 yr]` + electricity (measured
  `[12] Wh/req` × `[$0.17/kWh]`) + maintenance.
- **Cloud GPU (optional):** hourly rate × seconds/request.

![Break-even](../figures/break_even.png)

**Finding (example):** break-even ≈ `[230k]` requests/period. Below it the API
wins on cost; above it On-Prem wins. **Recommendation:** use the **API** for
low/spiky volume and rapid iteration; go **On-Prem** for high sustained volume
**or** when privacy, data security, and offline operation dominate — those can
justify local deployment even below the cost break-even. State assumptions
(prices, volume, hardware lifetime, tariff) so the analysis is reproducible.

---

## 7. Extensions (Task 5.7)

`[Describe your original extension: e.g. a QLoRA fine-tune pass and its effect on
TTFT/quality; a 7B-vs-14B-vs-32B size sweep; an input-length sweep quantifying
TTFT growth; or a new metric. Include the supporting figure.]`

---

## 8. Conclusions

A naive local run of a `[14B]` model is `[infeasible]` on our hardware; AirLLM
makes it run by paging layers from disk, and quantization makes it *bearable* by
cutting memory traffic — at a measured quality cost. Economically, local
deployment is rational `[above ~230k requests/period or when privacy/security
dominate]`. The decisive engineering insight: on modest hardware, large-model
inference is **memory- and disk-bound**, so the wins come from moving fewer bytes
(quantization) and from fast storage — not from raw compute.
