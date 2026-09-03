| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 64 | scale 1.53 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 2.9e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 512 | scale 1.53 | 1.00000 | 5.9e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 3.0e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace_prescaled | 64 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace_prescaled | 512 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | eager | reference_impl | 64 | scale 95.50 | — | — | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | chunked_m70_embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | chunked_m70_embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | mixed | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 5/5 |
| DeepSeek-V4-Flash-0731 | eager | effect_check | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 1.53 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 2.9e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 1.53 | 1.00000 | 5.9e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 3.0e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace_prescaled | 64 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace_prescaled | 512 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | reference_impl | 64 | scale 95.50 | — | — | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | graphs | chunked_m70_embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | chunked_m70_embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | mixed | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | effect_check | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes | checks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-V4-Flash-0731 | graphs | nosteer | 512 | 2.85 | 3.84 | -39.18 | 1.86 | 7,197 | 15 | — |
| DeepSeek-V4-Flash-0731 | graphs | nosteer | 1024 | 5.23 | 6.75 | 40.12 | 3.71 | 7,834 | 29 | — |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | 2.92 | 3.89 | 24.34 | 1.94 | 7,019 | 14 | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 1024 | 5.39 | 6.88 | 37.17 | 3.90 | 7,601 | 29 | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_add | 512 | 2.92 | 3.89 | 26.55 | 1.94 | 7,025 | 14 | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | embed_add | 1024 | 5.45 | 6.80 | 34.57 | 4.09 | 7,522 | 29 | 1/1 |