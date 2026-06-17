| Config | Prompt | Quant | TTFT (s) | TPOT (ms) | tok/s | Peak RAM (GB) | Peak VRAM (GB) | Energy (Wh) | Status |
|---|---|---|---|---|---|---|---|---|---|
| airllm-fp16 | short | fp16 | 122.61 | 128494.1 | 0.01 | 3.6 | 0.0 | 11.19 | ok |
| baseline | short | fp16 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0.00 | expected OOM ✓ (bottleneck confirmed) |
| ollama-q4 | long_context | q4 | 63.48 | 362.8 | 2.78 | 0.0 | 0.0 | 0.43 | ok |
| ollama-q4 | medium | q4 | 4.30 | 285.6 | 3.52 | 0.2 | 0.0 | 0.21 | ok |
| ollama-q4 | short | q4 | 39.51 | 255.4 | 4.12 | 0.1 | 0.0 | 0.18 | ok |
| ollama-q8 | short | q8 | 228.70 | 30161.3 | 0.03 | 0.4 | 0.0 | 3.34 | ok |
