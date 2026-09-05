| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| Qwen3-1.7B | compile | karvonen_add | 64 | coeff 1.0 | 0.99999 | 3.2e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0008 | 4/4 |
| Qwen3-1.7B | compile | karvonen_add | 512 | coeff 1.0 | 0.99999 | 3.5e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | compile | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.8e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0005 | 4/4 |
| Qwen3-1.7B | compile | karvonen_add | 512 | coeff 4.0 | 0.99999 | 2.0e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | compile | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.190 (clean-prompt noise 0.165); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | compile | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.151 (clean-prompt noise 0.165); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | compile | layer_replace | 64 | scale 23.13 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | compile | embed_replace | 64 | scale 1.38 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | embed_replace | 512 | scale 1.38 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | mixed | 64 | coeff 1.0 | 0.99999 | 3.2e-04 | 0.0e+00 | — | 5/5 |
| Qwen3.6-27B | compile | karvonen_add | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | karvonen_add | 512 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.8e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | layer_replace | 64 | scale 13.19 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | embed_replace | 64 | scale 0.97 | 1.00000 | 7.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | embed_replace | 512 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | mixed | 64 | coeff 1.0 | 0.99999 | 1.6e-04 | 0.0e+00 | — | 5/5 |

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes | checks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | compile | nosteer | 512 | 0.50 | 0.77 | 6.78 | 0.23 | 41,086 | 4 | — |
| Qwen3-1.7B | compile | nosteer | 1024 | 0.96 | 1.56 | 15.80 | 0.36 | 42,502 | 6 | — |
| Qwen3-1.7B | compile | karvonen_add | 512 | 0.53 | 0.81 | 7.37 | 0.24 | 38,999 | 4 | 1/1 (+1 n/a) |
| Qwen3-1.7B | compile | karvonen_add | 1024 | 1.04 | 1.68 | 15.54 | 0.40 | 39,228 | 5 | 1/1 (+1 n/a) |
| Qwen3-1.7B | compile | embed_replace | 512 | 0.55 | 0.82 | 11.66 | 0.28 | 37,255 | 3 | 1/1 (+1 n/a) |
| Qwen3-1.7B | compile | embed_replace | 1024 | 1.57 | 1.60 | 1.07 | 1.53 | 26,168 | 4 | 1/1 (+1 n/a) |