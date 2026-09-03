| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | eager | nosteer | 512 | 0.65 | 1.13 | 12.03 | 0.17 | 31,354 | 41 |
| Qwen3-1.7B | eager | nosteer | 1024 | 1.08 | 1.82 | 18.43 | 0.35 | 37,810 | 42 |
| Qwen3-1.7B | eager | karvonen_add | 512 | 0.66 | 1.14 | 12.06 | 0.18 | 31,044 | 41 |
| Qwen3-1.7B | eager | karvonen_add | 1024 | 1.15 | 2.16 | 25.39 | 0.13 | 35,676 | 41 |
| Qwen3-1.7B | eager | embed_replace | 512 | 0.67 | 1.16 | 12.20 | 0.19 | 30,377 | 41 |
| Qwen3-1.7B | eager | embed_replace | 1024 | 1.12 | 1.87 | 18.84 | 0.36 | 36,621 | 41 |
| Qwen3-1.7B | graphs | nosteer | 512 | 0.47 | 0.73 | 6.42 | 0.21 | 43,702 | 2 |
| Qwen3-1.7B | graphs | nosteer | 1024 | 0.90 | 1.48 | 14.53 | 0.32 | 45,548 | 3 |
| Qwen3-1.7B | graphs | karvonen_add | 512 | 0.50 | 0.75 | 6.33 | 0.24 | 41,224 | 2 |
| Qwen3-1.7B | graphs | karvonen_add | 1024 | 0.96 | 1.39 | 10.81 | 0.52 | 42,882 | 4 |
| Qwen3-1.7B | graphs | embed_replace | 512 | 0.49 | 0.74 | 6.23 | 0.24 | 41,641 | 2 |
| Qwen3-1.7B | graphs | embed_replace | 1024 | 0.92 | 1.40 | 12.01 | 0.44 | 44,518 | 3 |