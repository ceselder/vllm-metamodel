| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| Qwen3-1.7B | eager | karvonen_add | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0020 | 4/4 |
| Qwen3-1.7B | eager | karvonen_add | 512 | coeff 1.0 | 0.99999 | 2.8e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager | karvonen_add | 64 | coeff 4.0 | 0.99999 | 3.1e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0018 | 4/4 |
| Qwen3-1.7B | eager | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.5e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.135 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | eager | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.237 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | eager | layer_replace | 64 | scale 23.16 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager | embed_replace | 64 | scale 1.38 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | embed_replace | 512 | scale 1.38 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | mixed | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | — | 5/5 |
| Qwen3-1.7B | eager +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 3.1e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager +chunked | chunked_m10_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 4.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager +chunked | chunked_m70_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | karvonen_add | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0020 | 4/4 |
| Qwen3-1.7B | graphs | karvonen_add | 512 | coeff 1.0 | 0.99999 | 2.8e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs | karvonen_add | 64 | coeff 4.0 | 0.99999 | 3.1e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0018 | 4/4 |
| Qwen3-1.7B | graphs | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.5e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.135 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | graphs | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.237 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | graphs | layer_replace | 64 | scale 23.16 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs | embed_replace | 64 | scale 1.38 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | embed_replace | 512 | scale 1.38 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | mixed | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | — | 5/5 |
| Qwen3-1.7B | graphs +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 3.1e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs +chunked | chunked_m10_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 4.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs +chunked | chunked_m70_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | karvonen_add | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0001 | 4/4 |
| Qwen3.6-27B | eager | karvonen_add | 512 | coeff 1.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.4e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0002 | 4/4 |
| Qwen3.6-27B | eager | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.103 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | eager | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.121 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | eager | layer_replace | 64 | scale 13.19 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager | embed_replace | 64 | scale 0.97 | 1.00000 | 7.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | embed_replace | 512 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | mixed | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | — | 5/5 |
| Qwen3.6-27B | eager +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 1.4e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager +chunked | chunked_m10_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 2.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager +chunked | chunked_m70_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | karvonen_add | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0001 | 4/4 |
| Qwen3.6-27B | graphs | karvonen_add | 512 | coeff 1.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.4e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0002 | 4/4 |
| Qwen3.6-27B | graphs | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.103 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | graphs | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.121 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | graphs | layer_replace | 64 | scale 13.19 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs | embed_replace | 64 | scale 0.97 | 1.00000 | 7.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | embed_replace | 512 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | mixed | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | — | 5/5 |
| Qwen3.6-27B | graphs +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 1.4e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs +chunked | chunked_m10_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 2.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs +chunked | chunked_m70_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes | checks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | eager | nosteer | 512 | 0.65 | 1.13 | 11.14 | 0.17 | 31,354 | 41 | — |
| Qwen3-1.7B | eager | nosteer | 1024 | 1.08 | 1.82 | 18.43 | 0.35 | 37,810 | 42 | — |
| Qwen3-1.7B | eager | karvonen_add | 512 | 0.66 | 1.14 | 12.32 | 0.18 | 31,044 | 41 | 0/0 (+2 n/a) |
| Qwen3-1.7B | eager | karvonen_add | 1024 | 1.15 | 2.16 | 26.77 | 0.13 | 35,676 | 41 | 0/0 (+2 n/a) |
| Qwen3-1.7B | eager | embed_replace | 512 | 0.67 | 1.16 | 12.20 | 0.19 | 30,377 | 41 | 0/0 (+1 n/a) |
| Qwen3-1.7B | eager | embed_replace | 1024 | 1.12 | 1.87 | 18.84 | 0.36 | 36,621 | 41 | 0/0 (+1 n/a) |
| Qwen3-1.7B | graphs | nosteer | 512 | 0.47 | 0.73 | 6.42 | 0.21 | 43,702 | 2 | — |
| Qwen3-1.7B | graphs | nosteer | 1024 | 0.90 | 1.48 | 13.67 | 0.32 | 45,548 | 3 | — |
| Qwen3-1.7B | graphs | karvonen_add | 512 | 0.50 | 0.75 | 6.51 | 0.24 | 41,224 | 2 | 1/1 (+1 n/a) |
| Qwen3-1.7B | graphs | karvonen_add | 1024 | 0.96 | 1.39 | 12.79 | 0.52 | 42,882 | 4 | 1/1 (+1 n/a) |
| Qwen3-1.7B | graphs | embed_replace | 512 | 0.49 | 0.74 | 6.78 | 0.24 | 41,641 | 2 | 1/1 (+1 n/a) |
| Qwen3-1.7B | graphs | embed_replace | 1024 | 0.92 | 1.40 | 17.16 | 0.44 | 44,518 | 3 | 1/1 (+1 n/a) |
| Qwen3.6-27B | eager | nosteer | 512 | 7.70 | 12.48 | 119.42 | 2.92 | 2,660 | 82 | — |
| Qwen3.6-27B | eager | nosteer | 1024 | 13.27 | 20.69 | 185.38 | 5.86 | 3,086 | 123 | — |
| Qwen3.6-27B | eager | karvonen_add | 512 | 7.74 | 12.52 | 119.34 | 2.97 | 2,645 | 82 | 2/2 |
| Qwen3.6-27B | eager | karvonen_add | 1024 | 13.31 | 20.79 | 187.08 | 5.82 | 3,078 | 123 | 2/2 |
| Qwen3.6-27B | eager | embed_replace | 512 | 7.75 | 12.56 | 120.30 | 2.93 | 2,644 | 82 | 1/1 |
| Qwen3.6-27B | eager | embed_replace | 1024 | 13.31 | 20.78 | 186.74 | 5.84 | 3,078 | 123 | 1/1 |
| Qwen3.6-27B | graphs | nosteer | 512 | 5.66 | 8.46 | 69.88 | 2.87 | 3,617 | 4 | — |
| Qwen3.6-27B | graphs | nosteer | 1024 | 10.81 | 16.00 | 129.61 | 5.63 | 3,789 | 6 | — |
| Qwen3.6-27B | graphs | karvonen_add | 512 | 5.72 | 8.55 | 70.79 | 2.89 | 3,582 | 4 | 2/2 |
| Qwen3.6-27B | graphs | karvonen_add | 1024 | 10.84 | 16.10 | 131.44 | 5.58 | 3,779 | 6 | 2/2 |
| Qwen3.6-27B | graphs | embed_replace | 512 | 5.72 | 8.56 | 71.04 | 2.88 | 3,580 | 4 | 2/2 |
| Qwen3.6-27B | graphs | embed_replace | 1024 | 10.95 | 16.07 | 128.03 | 5.82 | 3,742 | 6 | 2/2 |