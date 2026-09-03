| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | eager | nosteer | 512 | 0.71 | 1.19 | 12.03 | 0.23 | 28,996 | 41 |
| Qwen3-1.7B | eager | nosteer | 1024 | 1.11 | 1.88 | 19.36 | 0.34 | 36,915 | 42 |
| Qwen3-1.7B | eager | karvonen_add | 512 | 0.71 | 1.21 | 12.67 | 0.20 | 29,046 | 41 |
| Qwen3-1.7B | eager | karvonen_add | 1024 | 1.20 | 1.94 | 18.45 | 0.46 | 34,152 | 42 |
| Qwen3-1.7B | eager | embed_replace | 512 | 0.73 | 1.22 | 12.43 | 0.23 | 28,226 | 41 |
| Qwen3-1.7B | eager | embed_replace | 1024 | 1.18 | 1.94 | 19.08 | 0.41 | 34,859 | 42 |
| Qwen3-1.7B | graphs | nosteer | 512 | 0.46 | 0.73 | 6.75 | 0.19 | 44,467 | 2 |
| Qwen3-1.7B | graphs | nosteer | 1024 | 0.89 | 1.39 | 12.45 | 0.39 | 46,145 | 3 |
| Qwen3-1.7B | graphs | karvonen_add | 512 | 0.50 | 0.77 | 6.76 | 0.23 | 41,054 | 2 |
| Qwen3-1.7B | graphs | karvonen_add | 1024 | 0.96 | 1.52 | 14.06 | 0.40 | 42,594 | 3 |
| Qwen3-1.7B | graphs | embed_replace | 512 | 0.50 | 0.77 | 6.80 | 0.23 | 41,093 | 2 |
| Qwen3-1.7B | graphs | embed_replace | 1024 | 0.98 | 1.54 | 14.00 | 0.42 | 41,744 | 3 |
| Qwen3.6-27B | eager | nosteer | 512 | 7.70 | 12.48 | 119.42 | 2.92 | 2,660 | 82 |
| Qwen3.6-27B | eager | nosteer | 1024 | 13.27 | 20.69 | 185.38 | 5.86 | 3,086 | 123 |
| Qwen3.6-27B | eager | karvonen_add | 512 | 7.74 | 12.52 | 119.34 | 2.97 | 2,645 | 82 |
| Qwen3.6-27B | eager | karvonen_add | 1024 | 13.31 | 20.79 | 187.08 | 5.82 | 3,078 | 123 |
| Qwen3.6-27B | eager | embed_replace | 512 | 7.75 | 12.56 | 120.30 | 2.93 | 2,644 | 82 |
| Qwen3.6-27B | eager | embed_replace | 1024 | 13.31 | 20.78 | 186.74 | 5.84 | 3,078 | 123 |
| Qwen3.6-27B | graphs | nosteer | 512 | 5.66 | 8.46 | 69.88 | 2.87 | 3,617 | 4 |
| Qwen3.6-27B | graphs | nosteer | 1024 | 10.81 | 16.00 | 129.61 | 5.63 | 3,789 | 6 |
| Qwen3.6-27B | graphs | karvonen_add | 512 | 5.72 | 8.55 | 70.79 | 2.89 | 3,582 | 4 |
| Qwen3.6-27B | graphs | karvonen_add | 1024 | 10.84 | 16.10 | 131.44 | 5.58 | 3,779 | 6 |
| Qwen3.6-27B | graphs | embed_replace | 512 | 5.72 | 8.56 | 71.04 | 2.88 | 3,580 | 4 |
| Qwen3.6-27B | graphs | embed_replace | 1024 | 10.95 | 16.07 | 128.03 | 5.82 | 3,742 | 6 |