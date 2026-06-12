---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/Nemotron-Labs-Diffusion-3B
library_name: peft
tags:
  - speculative-decoding
  - diffusion-language-model
  - lora
  - nemotron
  - inference-acceleration
  - drafter
---

# AeVox Diffusion Drafter — LoRA for Nemotron-Labs-Diffusion-3B

**Created by Daniel Rodd / AeVox.Ai.**

A LoRA adapter that **aligns NVIDIA's `Nemotron-Labs-Diffusion-3B`** to the token distribution of **`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`**, so the 3B can serve as a **cross-model speculative-decoding drafter** for the 120B.

## What it does
Speculative decoding accepts more tokens per expensive 120B forward when the drafter predicts the 120B's tokens well. This adapter raises the 3B drafter's **accepted tokens per target forward**:

| Drafter | accept / target-forward |
|---|---|
| Unaligned `Nemotron-Labs-Diffusion-3B` | 2.26 |
| 120B native MTP (baseline) | 2.75 |
| **+ this adapter** | **2.79** |

With an NVFP4-quantized drafter this projects to **~2.5× decode speedup** over autoregressive (acceptance measured offline; tok/s projected — see the repo's `docs/04-serving.md`).

## Usage
```python
import torch
from transformers import AutoModel
from peft import PeftModel

m = AutoModel.from_pretrained("nvidia/Nemotron-Labs-Diffusion-3B",
                              trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
m = PeftModel.from_pretrained(m, "Daniel-Rodd/aevox-diffusion-drafter-nemotron3-super").merge_and_unload()
# m.encoder / m.diffusion_head = the aligned drafter.
# See github.com/DRCubix/aevox-diffusion-drafter (src/eval_lora_acceptance.py) for the spec-decode loop.
```
Requires `transformers>=5.0` and `trust_remote_code=True`. The drafter and 120B share a byte-identical tokenizer (vocab 131,072), so draft token IDs are directly verifiable by the 120B.

## Training
- **Data:** 10K `(prompt, 120B-completion)` pairs (`DrCubix/nemotron3-super-120b-distill`), code-heavy, full-reasoning, temp=1.0.
- **Objective:** the 3B's native bidirectional diffusion CE on masked completion tokens.
- **LoRA:** rank 16 on `q/k/v/o + gate/up/down` (24.7M params, 0.64%), 3 epochs, lr 5e-5, eps-floored loss for stability.

## Limitations & honest notes
- Acceptance is **measured**; the ~2.5× speedup is **projected** (needs a custom vLLM drafter integration + NVFP4 quantization).
- The acceptance ceiling for this token-only, single-step, standalone drafter is ~2.8. Pushing to 3–4× needs hidden-state conditioning (DFlash) or tree drafting — see the repo's findings doc.

## Citation
See `CITATION.cff` in the repo. Created by **Daniel Rodd / AeVox.Ai**. Built on NVIDIA Nemotron models under the NVIDIA Open Model License.

**Repo:** https://github.com/DRCubix/aevox-diffusion-drafter
