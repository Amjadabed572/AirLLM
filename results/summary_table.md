| Config | Prompt | Quant | TTFT (s) | TPOT (ms) | tok/s | Total (s) | Peak RAM (GB) | Peak VRAM (GB) | Energy (Wh) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| airllm-fp16 | short | fp16 | 122.61 | 128494.1 | 0.01 | 2564.0 | 3.6 | 0.0 | 11.19 | ok |
| baseline | short | fp16 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0.0 | 0.00 | expected OOM ✓ (bottleneck confirmed) |
| ollama-q4 | long_context | q4 | 63.97 | 466.4 | 2.16 | 114.8 | 4.1 | 0.0 | 0.48 | ok |
| ollama-q4 | medium | q4 | 4.49 | 266.3 | 3.78 | 46.8 | 4.2 | 0.0 | 0.20 | ok |
| ollama-q4 | short | q4 | 52.13 | 295.7 | 3.56 | 57.7 | 4.2 | 0.0 | 0.24 | ok |
| ollama-q8 | short | q8 | 234.38 | 28139.7 | 0.04 | 769.0 | 5.0 | 0.0 | 3.20 | ok |
