| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| DeepSeek-V4-Flash-0731 | eager | mixed | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 5/5 |
| DeepSeek-V4-Flash-0731 | eager | effect_check | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 3/4 FAIL |