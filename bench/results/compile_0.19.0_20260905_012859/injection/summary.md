| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| Qwen3-1.7B | compile | karvonen_add | 64 | coeff 1.0 | 0.99999 | 3.5e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0010 | 4/4 |
| Qwen3-1.7B | compile | karvonen_add | 512 | coeff 1.0 | 0.99999 | 2.0e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | compile | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.2e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0007 | 4/4 |
| Qwen3-1.7B | compile | karvonen_add | 512 | coeff 4.0 | 0.99999 | 3.4e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | compile | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.151 (clean-prompt noise 0.128); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | compile | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.187 (clean-prompt noise 0.128); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | compile | layer_replace | 64 | scale 23.14 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | compile | embed_replace | 64 | scale 1.38 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | embed_replace | 512 | scale 1.38 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | compile | mixed | 64 | coeff 1.0 | 0.99999 | 3.5e-04 | 0.0e+00 | — | 5/5 |
| Qwen3.6-27B | compile | karvonen_add | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | karvonen_add | 512 | coeff 1.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.2e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.2e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | layer_replace | 64 | scale 13.19 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | compile | embed_replace | 64 | scale 0.97 | 1.00000 | 7.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | embed_replace | 512 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | compile | mixed | 64 | coeff 1.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 5/5 |

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes | checks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | compile | nosteer | 512 | 0.55 | 0.86 | 8.95 | 0.24 | 37,291 | 3 | — |
| Qwen3-1.7B | compile | nosteer | 1024 | 0.96 | 1.56 | 14.90 | 0.36 | 42,635 | 5 | — |
| Qwen3-1.7B | compile | karvonen_add | 512 | 0.56 | 0.88 | 8.59 | 0.24 | 36,555 | 3 | 1/1 (+1 n/a) |
| Qwen3-1.7B | compile | karvonen_add | 1024 | 1.08 | 1.65 | 13.23 | 0.51 | 37,903 | 5 | 2/2 |
| Qwen3-1.7B | compile | embed_replace | 512 | 0.63 | 0.93 | 7.53 | 0.34 | 32,350 | 3 | 1/1 (+1 n/a) |
| Qwen3-1.7B | compile | embed_replace | 1024 | 1.06 | 1.68 | 15.09 | 0.44 | 38,764 | 4 | 2/2 |