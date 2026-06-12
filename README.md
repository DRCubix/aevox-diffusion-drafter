<div align="center">

# AeVox Diffusion Drafter
### Aligning a diffusion language model as a cross-model speculative-decoding drafter for NVIDIA Nemotron-3-Super-120B

**Created by [Daniel Rodd](https://github.com/) · [AeVox.Ai](https://aevox.ai)**

</div>

---

> **TL;DR.** We LoRA-align NVIDIA's tri-mode **Nemotron-Labs-Diffusion-3B** to the outputs of **Nemotron-3-Super-120B-A12B** and use it as a *cross-model* speculative-decoding **drafter**. Alignment lifts the drafter's accepted-tokens-per-target-forward from **2.26 → 2.79**, beating the 120B's own built-in MTP self-speculation (**2.75**). With an NVFP4-quantized drafter this projects to **~2.5× decode speedup** over autoregressive. We also rigorously map the ceiling: data, epochs, multi-step sampling, and logit-distillation do **not** push past ~2.8 — the path to 3–4× is hidden-state conditioning (DFlash) or tree drafting.

This repo contains the **full, reproducible pipeline** (data generation → alignment training → acceptance evaluation), the trained LoRA adapter, and an honest writeup of the results and the ceiling.

## Why this exists
The 120B is served on a single **NVIDIA DGX Spark (GB10)** at ~22.9 tok/s using MTP speculative decoding (2.75 accept). We asked: *can a small, separately-trained **diffusion** model, aligned to the 120B's token distribution, draft for it better than MTP?* The answer: **yes at the acceptance level**, and this repo shows exactly how — plus where the limits are.

## Headline results (measured)
Accepted tokens per target forward (`nfe=1`, single-step diffusion draft), held-out:

| Drafter | accept / target-forward |
|---|---|
| Unaligned 3B diffusion | 2.26 |
| 120B native **MTP** (baseline to beat) | 2.75 |
| **Aligned 3B (this work)** | **2.79** |

**Projected decode speedup** (acceptance ÷ (1 + draft-cost ratio), see [docs/02-results.md](docs/02-results.md)):

| Drafter precision | draft/target cost `r` | projected speedup |
|---|---|---|
| bf16 | 0.55 | ~1.8× |
| fp8 | 0.23 | ~2.3× |
| **NVFP4** | 0.11 | **~2.5×** |

> **Honesty note.** Acceptance numbers are **measured offline** against the 120B's exact tokens (shared byte-identical tokenizer). The tok/s speedups are **projected** from the cost model; an end-to-end *served* benchmark needs a custom vLLM drafter integration (see [docs/04-serving.md](docs/04-serving.md), "Integration status"). We report what we measured and label what we projected.

## What's in here
```
src/        data-gen, capture, training (LoRA & KL-distill), evaluation
scripts/    serve the 120B (vLLM, MTP) + benchmark
docs/       full project layout, method, results, reproduction, findings, gotchas
model_card/ HuggingFace model card for the LoRA adapter
```

## Quickstart
```bash
pip install -r requirements.txt          # peft, transformers>=5.0, datasets
# 1. generate alignment data from the 120B (resumable, uploads to HF)
python3 src/build_distill.py prompts && python3 src/build_distill.py generate
# 2. LoRA-align the 3B diffusion drafter to the 120B outputs
python3 src/train_lora.py --epochs 3 --max_len 4096
# 3. evaluate acceptance vs MTP (2.75) / unaligned (2.26)
python3 src/eval_lora_acceptance.py --adapter ./lora_ckpt/epoch3
```
Full step-by-step (containers, hardware, gotchas) in [docs/03-reproduce.md](docs/03-reproduce.md).

## Artifacts
- **Model (LoRA adapter):** `Daniel-Rodd/aevox-diffusion-drafter-nemotron3-super` *(HuggingFace — see `model_card/`)*
- **Dataset (10K self-distillation):** `DrCubix/nemotron3-super-120b-distill`

## The honest ceiling (and the path to 3–4×)
We ruled out the cheap levers with evidence — more data, more epochs, multi-step sampling, and **logit-KL distillation** all plateau ~2.8 (the 120B's per-token distribution is too *peaked* for soft distillation to beat hard tokens). The bottleneck is the **mechanism**: a token-only, single-step, *standalone* drafter. Breaking past ~2.8 needs **hidden-state conditioning (DFlash)** or **tree drafting (DDTree + STree for the hybrid-Mamba target)**. Full analysis: [docs/05-findings-and-future-work.md](docs/05-findings-and-future-work.md).

## Credit & acknowledgements
**Created by Daniel Rodd / AeVox.Ai.** Please cite via [`CITATION.cff`](CITATION.cff).

Built on NVIDIA's open models — **Nemotron-3-Super-120B-A12B** and **Nemotron-Labs-Diffusion-3B** — used under the **NVIDIA Open Model License**. This project (the alignment method, code, LoRA adapter, and distilled dataset) is a derivative work; see [`LICENSE`](LICENSE). We thank the NVIDIA Nemotron-Labs team for releasing the tri-mode diffusion family.

## License
Code: **Apache-2.0** (© Daniel Rodd / AeVox.Ai). Model/data derivatives governed additionally by the **NVIDIA Open Model License**. See [`LICENSE`](LICENSE).
