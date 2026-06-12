# 02 — Results

All acceptance numbers are **accepted tokens per target forward** (the standard speculative-decoding metric), measured on held-out data against the 120B's exact token sequences. Higher is better. Baselines: **unaligned 3B = 2.26**, **120B native MTP = 2.75**.

## Headline: alignment beats MTP
Per-epoch held-out acceptance, `nfe=1` (single-step diffusion draft), 300-example eval:

| Stage | accept / target-forward |
|---|---|
| Unaligned 3B | 2.26 |
| MTP (target to beat) | 2.75 |
| Aligned 3B — epoch 1 | 2.76 |
| Aligned 3B — epoch 2 | 2.78 |
| **Aligned 3B — epoch 3** | **2.79** |

## Projected decode speedup
`speedup ≈ accept / (1 + nfe·r)`, `r = draft/target forward-time ratio`. Measured forward times: drafter 34 ms (bf16) vs target 62 ms.

| Drafter precision | `r` | speedup at accept=2.79 |
|---|---|---|
| bf16 | 0.55 | ~1.8× |
| fp8 | 0.23 | ~2.3× |
| **NVFP4** | 0.11 | **~2.5×** |

For reference, MTP delivers a **measured** ~1.41× / 22.9 tok/s on this GB10. The aligned drafter's projected ~2.5× (NVFP4) is a real improvement on the acceptance side; the *served* tok/s requires the drafter integration in [04-serving.md](04-serving.md).

## Ceiling map — what does NOT push past ~2.8
Each row is an experiment we ran to try to break the plateau:

| Lever | Result | Verdict |
|---|---|---|
| Alignment (LoRA, hard-CE) | 2.26 → 2.79 | ✅ beats MTP |
| More data (60 → 10,000 examples) | 2.72 → 2.76 (+0.04) | ❌ data-saturated |
| More epochs (1 → 3) | 2.76 → 2.79 (+0.03) | ❌ epoch-saturated |
| Multi-step nfe (threshold 0→0.9) | accept 2.30→2.57, **nfe 1→13** | ❌ speedup craters |
| Logit-KL distillation | 2.17 (≤ CE 2.30 on same eval) | ❌ peaked teacher |

### nfe sweep (aligned epoch-1 adapter, fast eval)
| K | threshold | accept | nfe | NVFP4 speedup |
|---|---|---|---|---|
| 16 | 0.0 | 2.30 | 1.0 | **2.06×** |
| 16 | 0.5 | 2.52 | 9.5 | 1.21× |
| 16 | 0.9 | 2.57 | 13.1 | 1.03× |

### Logit-KL head-to-head (40-ex / 200-tok, identical eval set)
| Model | accept |
|---|---|
| base (unaligned) | 2.00 |
| CE epoch1 | 2.26 |
| CE epoch3 | 2.30 |
| KL epoch1 | 2.17 |

## Interpretation
The bottleneck is the **mechanism**, not data/epochs/objective: a *token-only, single-step, standalone* 3B can only predict a *different* 120B's next tokens so well. Every information-poor lever saturates ~2.8. The levers that add new information — **hidden-state conditioning (DFlash)** and **tree drafting** — are the path to 3–4× (see [05-findings-and-future-work.md](05-findings-and-future-work.md)).

## Reproducibility
Every number above is produced by `src/eval_lora_acceptance.py`, `src/nfe_sweep.py`, and `src/fast_eval.py` on the checkpoints from `src/train_lora.py` / `src/train_kl.py`. See [03-reproduce.md](03-reproduce.md).
