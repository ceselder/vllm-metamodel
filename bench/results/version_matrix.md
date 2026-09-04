# vLLM version matrix (vllm-metamodels)

## Qwen/Qwen3.6-27B

| vLLM | fork stages | CPU tests | steering checks | injection checks | readout checks | plain vLLM (default) B=512 / 1024 | plain, V1 runner | fork steering + graphs B=512 / 1024 | fork steering eager B=512 | stock vllm-lens eager B=512 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.19.0 | 5/5 ok | ok | 20/20 | 20/20 | graphs: 0/0 | 5.23 s (3,913 tok/s) / 10.08 s (4,063 tok/s) | — / — | 5.51 s (3,714 tok/s) / 10.55 s (3,882 tok/s) | 7.81 s (2,621 tok/s) | n/a |
| 0.27.1 | 6/6 ok | ok | 20/20 | 18/20 | graphs: 0/0 | 5.02 s (4,083 tok/s) / 9.63 s (4,253 tok/s) | 5.05 s (4,058 tok/s) / 9.62 s (4,257 tok/s) | 6.15 s (3,332 tok/s) / 11.62 s (3,524 tok/s) | 7.74 s (2,645 tok/s) | n/a |

- vLLM 0.19.0 readout (graphs), per 1,024 texts: nocap 6.46 s, cap_all 10.38 s, cap_last5 6.74 s, read_last5 6.62 s, exit_read_last5 4.58 s
- vLLM 0.27.1 readout (graphs), per 1,024 texts: nocap 6.82 s, cap_all 11.89 s, cap_last5 7.06 s, read_last5 6.93 s, exit_read_last5 4.84 s
- vLLM 0.27.1 injection FAIL: graphs mixed: B=64 even requests (embed replace): downstream layers differ from clean only causally (positions < marker identical at layers 0 and L; marker row changed) pre-marker max|Δ| L0=4.88e-04 LL=6.25e-02; marker |Δ| L0≥1.181e+01
- vLLM 0.27.1 injection FAIL: graphs mixed: B=64 odd requests (karvonen add): delta == coeff·‖h‖·unit(v), embedding stream untouched, other rows untouched min cos=0.99999 ratio∈[0.9998,1.0002] other=6.2e-02 embed=0.0e+00

## Qwen/Qwen3-1.7B

| vLLM | fork stages | CPU tests | steering checks | injection checks | readout checks | plain vLLM (default) B=512 / 1024 | plain, V1 runner | fork steering + graphs B=512 / 1024 | fork steering eager B=512 | stock vllm-lens eager B=512 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.16.0 | 7/7 ok | ok | 61/61 | 50/50 | graphs: 0/0 | 0.62 s (33,209 tok/s) / 1.08 s (37,808 tok/s) | — / — | 0.65 s (31,672 tok/s) / 1.14 s (36,003 tok/s) | 1.15 s (17,851 tok/s) | 6.5 s @B=128 |
| 0.19.0 | 11/11 ok | ok | 61/61 | 142/142 | eager: 8/8, graphs: 8/8 | 0.56 s (36,429 tok/s) / 1.04 s (39,309 tok/s) | — / — | 0.66 s (31,236 tok/s) / 1.21 s (33,726 tok/s) | 1.10 s (18,579 tok/s) | 63.7 s |
| 0.28.0 | 7/7 ok | ok | 61/61 | 69/72 | graphs: 0/0 | 0.41 s (49,843 tok/s) / 0.74 s (55,505 tok/s) | — / — | 0.50 s (40,597 tok/s) / 0.91 s (44,894 tok/s) | 0.68 s (30,110 tok/s) | 5.3 s @B=128; 1.1.0: FAILS (V2 runner) |

- vLLM 0.16.0 readout (graphs), per 1,024 texts: nocap 0.54 s, cap_all 2.83 s, cap_last5 0.69 s, read_last5 0.68 s, exit_read_last5 0.77 s
- vLLM 0.19.0 readout (eager), per 1,024 texts: nocap 0.55 s, cap_all 1.97 s, cap_last5 0.66 s, read_last5 0.65 s, exit_read_last5 0.54 s
- vLLM 0.19.0 readout (graphs), per 1,024 texts: nocap 0.52 s, cap_all 2.03 s, cap_last5 0.66 s, read_last5 0.64 s, exit_read_last5 0.54 s
- vLLM 0.28.0 readout (graphs), per 1,024 texts: nocap 0.53 s, cap_all 1.31 s, cap_last5 0.59 s, read_last5 0.60 s, exit_read_last5 0.48 s
- vLLM 0.28.0 injection FAIL: eager embed_replace: B=512 norm_match=False: downstream layers differ from clean only causally (positions < marker within the clean-vs-clean noise floor at layers 0 and L; marker row changed) pre-marker max|Δ| L0=1.95e-03 LL=1.56e-02; marker |Δ| L0≥1.867e+00
- vLLM 0.28.0 injection FAIL: eager embed_replace: B=64 norm_match=True: downstream layers differ from clean only causally (positions < marker within the clean-vs-clean noise floor at layers 0 and L; marker row changed) pre-marker max|Δ| L0=3.91e-03 LL=3.12e-02; marker |Δ| L0≥1.891e+00
- vLLM 0.28.0 injection FAIL: eager embed_replace: B=512 norm_match=True: downstream layers differ from clean only causally (positions < marker within the clean-vs-clean noise floor at layers 0 and L; marker row changed) pre-marker max|Δ| L0=1.95e-03 LL=1.56e-02; marker |Δ| L0≥1.867e+00
