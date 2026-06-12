# 04 — Serving & Integration Status

## What works today (proven, measured)
Serving the 120B with its **native MTP** speculative decoding — the production baseline:
```bash
SPEC_TOKENS=3 MAX_BATCHED=8192 ./scripts/launch-nemotron-tuned.sh   # vLLM 26.05 container
./scripts/bench.sh 200                                              # ~22.9 tok/s on GB10 (1.41x over AR)
```
This gives an OpenAI-compatible endpoint on `:8000` (model `nemotron-3-super`). MTP accept ≈ 2.75.

## The aligned-drafter integration (status: NOT yet built)
**Honesty note (read before claiming served numbers):** the ~2.5× figure is **projected** from measured *acceptance* (2.79) and the NVFP4 drafter-cost model. To realize it as served tok/s, the aligned 3B must be wired into vLLM as a speculative-decoding **proposer**. That integration is **not in this repo**. Why, and what it takes:

- This vLLM build (`26.05`) exposes spec methods `draft_model, ngram, medusa, eagle/eagle3, mtp, dflash` — but **not** `custom_class`, and the 3B's architecture (`NemotronLabsDiffusionModel`) is **not in vLLM's model registry** (it's `trust_remote_code`-only). So neither `draft_model` nor `custom_class` can load it out of the box.
- Realizing it requires a **derived vLLM image** with a custom `SpecDecodeBaseProposer` subclass (a `NemotronDiffusionProposer`) that holds the 3B as an internal HF model and returns draft token IDs, plus registering a new method in the proposer dispatch. The Mamba2 state-rollback machinery is drafter-agnostic and is already exercised by the `mtp` path, so it is reusable — but you must run in the MTP-like regime and keep **Mamba prefix caching OFF**.

This is the clear next engineering step; the **trained adapter + the acceptance validation in this repo are the hard research part**, and they are done.

## Using the adapter for research / offline use today
You can load and run the aligned drafter directly to measure acceptance or do offline batched drafting:
```python
import torch
from transformers import AutoModel
from peft import PeftModel
m = AutoModel.from_pretrained("nvidia/Nemotron-Labs-Diffusion-3B", trust_remote_code=True,
                              dtype=torch.bfloat16).to("cuda").eval()
m = PeftModel.from_pretrained(m, "lora_ckpt/epoch3").merge_and_unload()
# m.encoder / m.diffusion_head are now the aligned drafter; see src/eval_lora_acceptance.py
```

## Quantizing the drafter (the speed lever)
The 3B ships **bf16 only**. To realize the NVFP4 `r≈0.11` projection, quantize the *merged* aligned drafter to NVFP4 (or fp8) for inference — recommended via **NVFP4 QAT (NVIDIA TensorRT Model Optimizer)** rather than naive PTQ, to preserve acceptance. (NVIDIA themselves pretrained Nemotron-3 with NVFP4 mixed-precision training; the drafter is small enough that QAT for deployment is the appropriate, low-risk path.)

## Deploying for "all my devices"
- The **simplest deployable artifact today** is the 120B + MTP via `scripts/launch-nemotron-tuned.sh` (proven 22.9 tok/s) — run it on each GB10-class device.
- The **aligned drafter** ships as a LoRA adapter (`model_card/`) + this pipeline; deploying it for higher tok/s is gated on the proposer integration above.
