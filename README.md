# EX05 — Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

> Course L08 · Dr. Yoram Segal · June 2026
> A reproducible, instrumented experiment that runs a large language model
> **on-premises** on modest hardware using **AirLLM** layer-streaming and
> **quantization**, then analyses the result both technically and economically.

**Hebrew version:** see [`README.he.md`](README.he.md).
**Full technical report:** [`reports/report.md`](reports/report.md) (EN) ·
[`reports/report.he.md`](reports/report.he.md) (HE).

> ⚠️ **The numbers and figures committed here are an EXAMPLE template** that
> proves the pipeline runs. Replace them with *your own measured results* by
> running the steps in [Reproduce](#reproduce). Every figure regenerates
> automatically from `results/*.json`.

---

## 1. Hardware spec & model choice (Task 5.1)

Generated automatically by `python -m src.hardware` → `results/hardware.json`.
Paste your real output here:

| Field | Value |
|---|---|
| OS | _e.g._ Windows 11 |
| CPU | _e.g._ Intel i7-12700H (14 cores) |
| RAM | _e.g._ 16 GB |
| GPU / VRAM | _e.g._ RTX 3060 Laptop / 6 GB (or None) |
| Disk free / type | _e.g._ 380 GB free, NVMe SSD |
| Python | 3.10–3.12 |

**Model chosen:** `Qwen/Qwen2.5-14B-Instruct` (example).
**Why:** Big enough to make a naive full load fail on 16 GB RAM (so the
bottleneck is real and demonstrable), but its shards fit on an NVMe drive and a
single transformer layer fits in RAM — exactly the regime where AirLLM earns its
keep. The `src.model_selector` helper recommends a tier from your detected RAM
and free disk; **justify your final pick in the report.**

---

## 2. What the experiment does

1. **Baseline (Task 5.2)** — load the model the naive way
   (`transformers.AutoModelForCausalLM`, full weights). On a "big enough" model
   this OOMs / thrashes swap / crawls. That failure is the bottleneck evidence.
2. **AirLLM + quantization (Task 5.3)** — the same prompt through AirLLM, which
   streams **one layer at a time** from disk, swept across **FP16 → Q8 → Q4**.
3. **Measure (Task 5.4)** — TTFT, ITL/TPOT, throughput, peak RAM/VRAM, energy.
4. **Economics (Task 5.5)** — On-Prem CAPEX/OPEX vs third-party API, with
   optional cloud-GPU and prompt-caching scenarios; find the break-even volume.

Measurement tools: a background `MemorySampler` thread (peak RSS + CUDA peak),
a streaming timer for TTFT/per-token gaps, and `matplotlib` for all figures.

---

## 3. Results summary (EXAMPLE — replace with yours)

`python -m analysis.analyze` writes `results/summary_table.md` and the figures.

| Config | Quant | TTFT (s) | TPOT (ms) | tok/s | Peak RAM (GB) | Energy (Wh) | Status |
|---|---|---|---|---|---|---|---|
| baseline | fp16 | — | — | — | — | — | **FAILED (OOM)** |
| airllm | fp16 | 9.80 | 1850 | 0.53 | 7.9 | 38.0 | ok |
| airllm | q8 | 7.10 | 1200 | 0.81 | 6.1 | 24.0 | ok |
| airllm | q4 | 5.30 | 780 | 1.24 | 4.4 | 14.0 | ok |

**Reading it:** the naive baseline cannot even load — AirLLM trades RAM for disk
I/O to make it run at all. Quantization (Q4) roughly **halves peak RAM and TPOT**
versus FP16, at some output-quality cost (the "red line" of accuracy).

![Throughput](figures/throughput.png)
![Prefill vs Decode](figures/ttft_vs_tpot.png)
![Peak RAM](figures/peak_ram.png)

---

## 4. Inference-concept analysis (Task 5.6)

- **Prefill = TTFT.** Building the KV-cache over the prompt is one big parallel
  matmul → **compute-bound**. Watch TTFT grow with the `long_context` prompt.
- **Decode = TPOT.** Each new token streams the whole model's weights through
  memory once → **memory-bound**. TPOT is dominated by memory bandwidth, which
  is why AirLLM's per-layer disk reads hurt here most.
- **AirLLM ≈ OS virtual memory / paging**, but for *model layers*: only the
  "page" (layer) you need is resident; the rest lives on disk. The disk
  (`SSD/NVMe I/O`) becomes the true bottleneck, not RAM size.
- **Quantization** shrinks each weight (FP16→Q4 ≈ 4× smaller), cutting the bytes
  moved per token → faster decode and lower peak memory, until accuracy degrades.

![Roofline](figures/roofline.png)

---

## 5. Economic analysis & recommendation (Task 5.5)

![Break-even](figures/break_even.png)

Two transparent cost models in [`analysis/economics.py`](analysis/economics.py),
all assumptions editable:

- **API:** `requests × (in·price_in + out·price_out)`, with optional
  **prompt-caching** discount on the repeated prefix (PagedAttention-style
  providers charge far less for cached tokens — this *shifts the break-even*).
- **On-Prem:** amortized hardware CAPEX + electricity (from measured Wh/req) +
  maintenance.
- **Optional cloud GPU:** hourly rate × seconds/request.

**Example finding:** break-even ≈ **230k requests** over the period. Below that,
the API wins on pure cost; above it, On-Prem wins — *before* accounting for
privacy, data security, and offline availability, which can favour On-Prem
regardless of volume. **State your real prices and cite the provider.**

---

## 6. Extensions (Task 5.7)

Document at least one original extension here — e.g. a LoRA/QLoRA fine-tune step,
a multi-model size comparison, an added metric, or an input-length sweep showing
TTFT scaling. The `long_context` prompt is wired in to support the last one.

---

## Reproduce

```bash
# 1. Environment (uv only)
uv venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e .

# 2. Document hardware + see the recommended model
python -m src.hardware
python -m src.model_selector

# 3. (optional) set a fast, non-OS drive for the shards
#    Windows:  set AIRLLM_SHARDS=D:\airllm_shards
#    Linux:    export AIRLLM_SHARDS=/fast/airllm_shards
#    or edit experiments/config.py

# 4. Run experiments  (baseline first, then the AirLLM sweep)
python -m experiments.run_baseline
python -m experiments.run_airllm

# 5. Build tables + figures + economics
python -m analysis.analyze
```

### Known pitfalls (from the assignment's Do/Don't list)
- **Python 3.13 is too new** for AirLLM/bitsandbytes — pin 3.10–3.12.
- **Point `layer_shards_saving_path` at a fast non-OS drive** — shards are tens
  of GB of SafeTensors and flood `C:` otherwise.
- Use **`airllm.AutoModel`** (the general class) to avoid the class-mismatch
  error with `Qwen`-family models.
- **Never commit your Hugging Face token** — it's git-ignored here.
- Start small (Q4, low `max_new_tokens`) to confirm the "pipe" works, then scale.

---

## Repository layout

```
README.md / README.he.md     project docs (this file)
pyproject.toml               uv project + deps
src/                         hardware, metrics, model selector, runners, prompts
experiments/                 config + run_baseline + run_airllm
analysis/                    economics, plots, analyze
results/                     raw per-run JSON (kept) + summary_table.md
figures/                     generated PNGs (embedded above)
reports/                     report.md (EN) + report.he.md (HE)
```
