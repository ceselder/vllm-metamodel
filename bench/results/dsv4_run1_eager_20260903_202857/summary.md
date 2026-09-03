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
| DeepSeek-V4-Flash-0731 | eager | mixed | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/5 FAIL |
| DeepSeek-V4-Flash-0731 | eager | chunked_m70_embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | chunked_m70_embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |