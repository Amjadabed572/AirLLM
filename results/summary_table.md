| Config | Prompt | Quant | TTFT (s) | TPOT (ms) | tok/s | Peak RAM (GB) | Peak VRAM (GB) | Energy (Wh) | Status |
|---|---|---|---|---|---|---|---|---|---|
| airllm-fp16 | — | fp16 | 129.48 | 144190.5 | 0.01 | 3.6 | 0.0 | 12.49 | ok |
| baseline | — | fp16 | 0.00 | 0.0 | 0.00 | 0.0 | 0.0 | 0.00 | FAILED — Process killed by OS during naive in-RAM load (hard OOM) |
| ollama-q4 | long_context | q4 | 57.06 | 286.6 | 3.52 | 0.0 | 0.0 | 0.37 | ok |
| ollama-q4 | medium | q4 | 3.83 | 271.3 | 3.71 | 0.3 | 0.0 | 0.20 | ok |
| ollama-q4 | short | q4 | 44.44 | 245.1 | 4.29 | 1.8 | 0.0 | 0.20 | ok |
| ollama-q8 | — | q8 | 272.43 | 30546.0 | 0.03 | 1.8 | 0.0 | 3.55 | ok |
