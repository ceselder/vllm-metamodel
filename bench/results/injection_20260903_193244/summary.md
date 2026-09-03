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

| model | engine | condition | B | wall (s) | tok/s | hook passes |
|---|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | eager | nosteer | 512 | 1.49 | 13,721 | 41 |
| Qwen3-1.7B | eager | nosteer | 1024 | 1.37 | 29,808 | 41 |
| Qwen3-1.7B | eager | karvonen_add | 512 | 0.96 | 21,303 | 41 |
| Qwen3-1.7B | eager | karvonen_add | 1024 | 1.50 | 27,371 | 41 |
| Qwen3-1.7B | eager | embed_replace | 512 | 1.01 | 20,373 | 41 |
| Qwen3-1.7B | eager | embed_replace | 1024 | 1.50 | 27,369 | 42 |
| Qwen3-1.7B | graphs | nosteer | 512 | 0.51 | 39,879 | 2 |
| Qwen3-1.7B | graphs | nosteer | 1024 | 0.95 | 43,264 | 3 |
| Qwen3-1.7B | graphs | karvonen_add | 512 | 0.58 | 35,382 | 2 |
| Qwen3-1.7B | graphs | karvonen_add | 1024 | 1.10 | 37,399 | 2 |
| Qwen3-1.7B | graphs | embed_replace | 512 | 0.57 | 35,799 | 2 |
| Qwen3-1.7B | graphs | embed_replace | 1024 | 1.01 | 40,598 | 2 |
| Qwen3.6-27B | eager | nosteer | 512 | 5.91 | 3,467 | 82 |
| Qwen3.6-27B | eager | nosteer | 1024 | 10.50 | 3,902 | 123 |
| Qwen3.6-27B | eager | karvonen_add | 512 | 5.87 | 3,487 | 82 |
| Qwen3.6-27B | eager | karvonen_add | 1024 | 10.58 | 3,872 | 123 |
| Qwen3.6-27B | eager | embed_replace | 512 | 5.87 | 3,492 | 82 |
| Qwen3.6-27B | eager | embed_replace | 1024 | 10.60 | 3,864 | 123 |
| Qwen3.6-27B | graphs | nosteer | 512 | 5.38 | 3,807 | 4 |
| Qwen3.6-27B | graphs | nosteer | 1024 | 10.31 | 3,971 | 6 |
| Qwen3.6-27B | graphs | karvonen_add | 512 | 5.43 | 3,775 | 4 |
| Qwen3.6-27B | graphs | karvonen_add | 1024 | 10.32 | 3,968 | 6 |
| Qwen3.6-27B | graphs | embed_replace | 512 | 5.45 | 3,760 | 4 |
| Qwen3.6-27B | graphs | embed_replace | 1024 | 10.35 | 3,956 | 6 |