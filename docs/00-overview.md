# 00 — Project Overview

**AeVox Diffusion Drafter** — created by **Daniel Rodd / AeVox.Ai**.

## The idea in one paragraph
Speculative decoding accelerates a large autoregressive (AR) LLM by having a cheap **drafter** propose several tokens that the large model **verifies** in a single forward pass. NVIDIA's `Nemotron-3-Super-120B-A12B` ships with **MTP** (multi-token prediction) self-speculation that accepts ~2.75 tokens per forward. We ask whether a *separate, small **diffusion** language model* — NVIDIA's tri-mode `Nemotron-Labs-Diffusion-3B` — can be **LoRA-aligned to the 120B's token distribution** and used as a **cross-model drafter** that beats MTP. It does, at the acceptance level: **2.79 vs 2.75**, up from **2.26** unaligned.

## Why a diffusion drafter?
A diffusion LM fills a whole **block** of masked tokens **in parallel** in one forward — a natural fit for proposing K draft tokens at once. The drafter and the 120B share a **byte-identical tokenizer (vocab 131,072)**, so draft token IDs are directly verifiable by the 120B (the binding requirement for speculative decoding).

## System at a glance
```
                 prompt
                   │
                   ▼
   ┌─────────────────────────────┐        ┌────────────────────────────┐
   │  Drafter: Nemotron-Labs-     │ block  │  Target: Nemotron-3-Super- │
   │  Diffusion-3B  +  LoRA(align) │──────▶ │  120B-A12B  (verify, 1 fwd)│
   │  (1 diffusion forward → K)    │ of K   │  hybrid Mamba2 + MoE, NVFP4│
   └─────────────────────────────┘ tokens  └────────────────────────────┘
                   ▲                                   │ accept longest match
                   └───────────────────────────────────┘  + 1 bonus token
```

## Hardware this was built on
- **ASUS DGX Spark**, NVIDIA **GB10** (Grace Blackwell, sm_121, arm64), **128 GB unified memory**, CUDA 13 / driver 595.71.05.
- Single GPU — training and serving contend for it. Everything runs in the `nvcr.io/nvidia/vllm:26.05-py3` container (vLLM 0.20.1).

## Repository layout
| Path | Contents |
|---|---|
| `src/build_distill.py` | Generate the self-distillation dataset from the 120B (prompts / generate / probe) |
| `src/capture_logits.py` | Teacher-forced top-k logit capture (for KL-distillation experiments) |
| `src/train_lora.py` | LoRA-align the 3B drafter (hard-token diffusion objective) + `--resume_adapter` |
| `src/train_kl.py` | Logit-KL distillation variant (Exp1) |
| `src/eval_lora_acceptance.py` | Acceptance eval (accepted tokens/forward) — importable + CLI |
| `src/nfe_sweep.py` | Sweep denoising steps (threshold→nfe) + projected speedup |
| `src/fast_eval.py` | Fast head-to-head of base/CE/KL adapters |
| `src/qa_dataset.py` | Dataset quality checks |
| `scripts/launch-nemotron-tuned.sh` | Serve the 120B with MTP (the 22.9 tok/s baseline) |
| `scripts/bench.sh` | tok/s benchmark against the served endpoint |
| `docs/` | This documentation set |
| `model_card/` | HuggingFace model card for the LoRA adapter |

## Documents
- [01-method.md](01-method.md) — how alignment + diffusion drafting works
- [02-results.md](02-results.md) — measured acceptance, the speedup model, all numbers
- [03-reproduce.md](03-reproduce.md) — end-to-end reproduction
- [04-serving.md](04-serving.md) — serving the 120B + drafter integration status
- [05-findings-and-future-work.md](05-findings-and-future-work.md) — the ceiling analysis and the path to 3–4×
- [06-gotchas.md](06-gotchas.md) — hard-won operational notes (GB10 unified memory, tokenizer quirks, etc.)
