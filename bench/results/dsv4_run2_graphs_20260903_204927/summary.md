| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 1.53 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 2.9e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 1.53 | 1.00000 | 5.9e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 3.0e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace_prescaled | 64 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace_prescaled | 512 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | reference_impl | 64 | scale 95.50 | — | — | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | graphs | mixed | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/5 FAIL |
| DeepSeek-V4-Flash-0731 | graphs | effect_check | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 3/4 FAIL |
| DeepSeek-V4-Flash-0731 | graphs | chunked_m70_embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | chunked_m70_embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes | checks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-V4-Flash-0731 | graphs | nosteer | 512 | 2.85 | 3.82 | -288.01 | 1.89 | 7,179 | 14 | — |
| DeepSeek-V4-Flash-0731 | graphs | nosteer | 1024 | 5.25 | 6.75 | 36.71 | 3.75 | 7,802 | 29 | — |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | 2.93 | 3.91 | 56.21 | 1.96 | 6,987 | 14 | 1/2 FAIL |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 1024 | 5.43 | 6.88 | 32.86 | 3.99 | 7,538 | 29 | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_add | 512 | 2.93 | 3.91 | 56.13 | 1.96 | 6,981 | 14 | 0/1 FAIL |
| DeepSeek-V4-Flash-0731 | graphs | embed_add | 1024 | 5.45 | 6.94 | 8.42 | 3.96 | 7,516 | 29 | 1/1 |