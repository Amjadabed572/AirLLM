# EX05 — Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

> Course L08 · Dr. Yoram Segal · June 2026
> A reproducible, instrumented experiment that runs a large language model
> **on-premises** on a modest 8 GB laptop using **AirLLM** layer-streaming, plus
> a **GGUF quantization** comparison via Ollama, analysed both technically and
> economically.

**Hebrew version:** see [`README.he.md`](README.he.md).
**Full technical report:** [`reports/report.md`](reports/report.md) (EN) ·
[`reports/report.he.md`](reports/report.he.md) (HE).

> ⏳ **Status of the numbers below.** Hardware specs (`results/hardware.json`)
> and the economic / roofline figures are **real and committed**. The
> performance tables (TTFT/TPOT/throughput/RAM) are marked **PENDING** until the
> model runs are executed on the target machine — see [Reproduce](#reproduce).
> No invented measurements are committed; every performance figure regenerates
> from `results/*.json` via `python -m analysis.analyze`.

---

## 1. Hardware spec & model choice (Task 5.1)

Auto-detected by `python -m src.hardware` → [`results/hardware.json`](results/hardware.json):

| Field | Value (this machine) |
|---|---|
| OS | Windows 10 |
| CPU | Intel Core i7-7500U @ 2.70 GHz — **2 physical / 4 logical cores** |
| RAM | **7.9 GB** total |
| GPU / VRAM | NVIDIA GeForce GTX 950M / **2.0 GB** (Maxwell, compute 5.0) |
| Disk | **SSD** (Micron 1100 SATA), ~40 GB free |
| Python | 3.12 (uv-managed; pinned `>=3.10,<3.13`) |

**Model chosen:** `Qwen/Qwen2.5-7B-Instruct`.
**Why this is the right "big enough to hurt, not impossible" pick *for this machine*:**
its FP16 footprint (~15 GB) **far exceeds the 7.9 GB RAM**, so the naive baseline
is guaranteed to fail (the bottleneck is real and demonstrable). Yet its
per-layer shards fit the ~40 GB SSD and a single transformer layer fits in RAM —
exactly the regime where AirLLM earns its keep. A 14B/32B model would not fit the
free disk once shards are written; 72B is far out of reach. `src.model_selector`
reproduces this reasoning from the detected RAM + free disk.

---

## 2. What the experiment does

1. **Baseline (Task 5.2)** — load the model the naive way
   (`transformers.AutoModelForCausalLM`, full FP16 weights). On 7.9 GB RAM this
   OOMs / thrashes swap. **That failure is the bottleneck evidence**, captured
   verbatim in `results/baseline_*.json`.
2. **AirLLM (Task 5.3)** — the same prompt through AirLLM, which streams **one
   transformer layer at a time** from the SSD. This is the FP16 layer-streaming
   demonstration that makes the otherwise-impossible model run.
3. **Quantization comparison (Task 5.3)** — **Q4 vs Q8 GGUF via Ollama**
   (llama.cpp, CPU). *Why not AirLLM's own q4/q8?* That path needs `bitsandbytes`
   with a CUDA GPU of compute capability ≥ 7.5 and several GB VRAM; the GTX 950M
   (Maxwell 5.0, 2 GB) cannot run it — a documented hardware limitation. GGUF on
   CPU gives the real quantization numbers instead.
4. **Measure (Task 5.4)** — TTFT, ITL/TPOT, throughput, peak RAM/VRAM, energy.
5. **Economics (Task 5.5)** — On-Prem CAPEX/OPEX vs third-party API, with
   optional cloud-GPU and prompt-caching scenarios; compute the break-even volume.

Measurement tools: a background `MemorySampler` thread (peak RSS + CUDA peak), a
streaming timer for TTFT/per-token gaps, Ollama's native ns-precision counters
for the GGUF runs, and `matplotlib` for all figures.

---

## 3. Results summary

`python -m analysis.analyze` writes `results/summary_table.md` and the figures
from whatever real runs exist in `results/`.

| Config | Quant | TTFT (s) | TPOT (ms) | tok/s | Peak RAM (GB) | Energy (Wh) | Status |
|---|---|---|---|---|---|---|---|
| baseline (HF direct) | fp16 | — | — | — | — | — | **PENDING run** (expected: FAILED/OOM) |
| airllm | fp16 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | PENDING run |
| ollama (GGUF) | q8 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | PENDING run |
| ollama (GGUF) | q4 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | PENDING run |

> Replace this table with the auto-generated `results/summary_table.md` after the
> runs complete. **Expected shape of the result** (to be confirmed by data, not
> assumed): the baseline cannot load at all; AirLLM runs but is heavily
> disk-I/O-bound (seconds-per-token); Q4 < Q8 for peak RAM and TPOT, at some
> output-quality cost — the accuracy "red line".

<!-- These regenerate after model runs:
![Throughput](figures/throughput.png)
![Prefill vs Decode](figures/ttft_vs_tpot.png)
![Peak RAM](figures/peak_ram.png)
-->

---

## 4. Inference-concept analysis (Task 5.6)

- **Prefill = TTFT.** Building the KV-cache over the prompt is one big parallel
  matmul → **compute-bound**. Watch TTFT grow with the `long_context` prompt.
- **Decode = TPOT.** Each new token streams the whole model's weights through
  memory once → **memory-bound**. With AirLLM those weights come from the *SSD*,
  so decode is bound by disk read bandwidth — the dominant cost here.
- **AirLLM ≈ OS virtual memory / paging**, but for *model layers*: only the
  "page" (layer) you need is resident; the rest lives on disk. The disk becomes
  the true bottleneck, not RAM size.
- **Quantization** shrinks each weight (FP16→Q4 ≈ 4× smaller), cutting the bytes
  moved per token → faster decode and lower peak memory, until accuracy degrades.

![Roofline](figures/roofline.png)

---

## 5. Economic analysis & recommendation (Task 5.5)

![Break-even](figures/break_even.png)

Two transparent cost models in [`analysis/economics.py`](analysis/economics.py),
all assumptions editable and stated:

- **API:** `requests × (in·price_in + out·price_out)`, with optional
  **prompt-caching** discount on the repeated prefix (providers charge far less
  for cached prefix tokens — this *shifts the break-even rightward*).
- **On-Prem:** amortized hardware CAPEX + electricity (from the **measured**
  Wh/request) + maintenance.
- **Optional cloud GPU:** hourly rate × seconds/request.

With the committed default assumptions the computed break-even is
**≈ 230k requests** over the amortization period: below that the API wins on pure
cost; above it On-Prem wins — *before* accounting for privacy, data security, and
offline availability, which can favour On-Prem regardless of volume. **Edit the
prices/tariff/lifetime in `economics.py` to your cited values; the energy/request
must come from your measured runs.**

---

## 6. Extensions (Task 5.7)

This project's required original extension is the **GGUF-via-Ollama quantization
track** (§2.3): it makes a real Q4/Q8 comparison achievable on a GPU that cannot
run bitsandbytes, and contrasts two fundamentally different local-inference
engines (AirLLM disk-streaming vs llama.cpp CPU quantization). The
`long_context` prompt additionally supports a TTFT-vs-input-length sweep.

---

## Reproduce

```powershell
# 1. Environment (uv). Windows PowerShell shown; bash is analogous.
uv venv
uv pip install -e .                      # heavy: torch, transformers, airllm, ...

# 2. Document hardware + see the recommended model (already committed, real)
.\.venv\Scripts\python.exe -m src.hardware
.\.venv\Scripts\python.exe -m src.model_selector

# 3. Point AirLLM shards at a drive with room (defaults to .\layer_shards)
#    PowerShell:  $env:AIRLLM_SHARDS = "D:\airllm_shards"
#    or edit experiments/config.py

# 4. Run experiments
.\.venv\Scripts\python.exe -m experiments.run_baseline     # expect OOM = the bottleneck
.\.venv\Scripts\python.exe -m experiments.run_airllm       # FP16 layer-streaming

# 5. Quantization comparison via Ollama/GGUF (install Ollama app first)
uv pip install ollama
.\.venv\Scripts\python.exe -m experiments.run_ollama       # Q4 + Q8 on CPU

# 6. Build tables + figures + economics
.\.venv\Scripts\python.exe -m analysis.analyze
```

### Known pitfalls (from the assignment's Do/Don't list)
- **Python 3.13 is too new** for AirLLM/bitsandbytes — this project pins
  `>=3.10,<3.13` (uv selects 3.12).
- **transformers must stay `<4.45`.** AirLLM 2.11 hard-imports
  `optimum.bettertransformer`, which breaks on transformers 5.x (removed
  `is_tf_available`). `pyproject.toml` pins `transformers>=4.44,<4.45`,
  `optimum<2.0`, and `sentencepiece` (needed by an AirLLM tokenizer import). If
  you ever see `cannot import name 'is_tf_available'` or
  `No module named 'optimum.bettertransformer'`, your transformers/optimum drifted
  too new — reinstall with `uv pip install -e .`.
- **Point `layer_shards_saving_path` at a drive with space** — shards are ~15 GB
  of SafeTensors for 7B and flood `C:` otherwise. Only ~40 GB is free here, so
  watch disk during the AirLLM run.
- Use **`airllm.AutoModel`** (the general class) to avoid the class-mismatch
  error with `Qwen`-family models.
- **Never commit your Hugging Face token** — `.gitignore` excludes tokens, the
  venv, model shards, and `*.safetensors`/`*.gguf`.
- Start small (low `max_new_tokens`) to confirm the "pipe" works, then scale.
- **Be patient:** on this 8 GB / SATA-SSD machine, AirLLM reads ~15 GB from disk
  *per token*; expect tens of seconds per token. Keep `max_new_tokens` modest.

---

## Repository layout

```
README.md / README.he.md     project docs (this file)
pyproject.toml               uv project + deps
src/                         hardware, metrics, model selector, runners, prompts
                             (baseline_runner, airllm_runner, ollama_runner)
experiments/                 config + run_baseline + run_airllm + run_ollama
analysis/                    economics, plots, analyze
results/                     hardware.json (real) + raw per-run JSON + summary_table.md
figures/                     generated PNGs (economics/roofline real; perf pending)
reports/                     report.md (EN) + report.he.md (HE)
```
