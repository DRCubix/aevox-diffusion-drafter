#!/usr/bin/env python3
"""
Sweep nfe (via threshold/K) on the LoRA-aligned epoch-1 drafter to see whether
multi-step diffusion drafting unlocks higher acceptance / 3-4x. Loads the adapter
once. Reports accept, nfe, and projected speedup at bf16/fp8/NVFP4 drafter cost.
"""
import torch
from transformers import AutoModel
from peft import PeftModel
from eval_lora_acceptance import acceptance_eval, _load_rows

DIFF_3B = "nvidia/Nemotron-Labs-Diffusion-3B"
ADAPTER = "/work/lora_ckpt/epoch1"
TGT_MS = 61.7   # 120B target forward (ms)
DRAFT_MS = {"bf16": 34.0, "fp8": 14.0, "nvfp4": 7.0}  # 3B draft forward by precision

print("loading base + epoch1 adapter...", flush=True)
m = AutoModel.from_pretrained(DIFF_3B, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
m = PeftModel.from_pretrained(m, ADAPTER).merge_and_unload()

val = _load_rows("/work/distill_shards/shard_00009.jsonl", limit=25)
for r in val:                                  # cap sim length — acceptance ~stationary; keeps sweep fast
    r["completion_ids"] = r["completion_ids"][:200]
configs = [(16, 0.0), (16, 0.5), (16, 0.9)]    # nfe=1 baseline + two multi-step points

print(f"\n=== nfe sweep on aligned epoch1 ({len(val)} held-out) — vs MTP 2.75 ===", flush=True)
print(f"{'K':>3} {'thr':>4} {'accept':>7} {'nfe':>6}  | speedup  bf16  fp8  nvfp4", flush=True)
for K, thr in configs:
    acc, nfe = acceptance_eval(m, val, K=K, threshold=thr, device="cuda", max_eval=len(val))
    sp = {p: acc / (1 + nfe * DRAFT_MS[p] / TGT_MS) for p in DRAFT_MS}
    print(f"{K:>3} {thr:>4} {acc:>7.2f} {nfe:>6.2f}  |          "
          f"{sp['bf16']:>4.2f}x {sp['fp8']:>4.2f}x {sp['nvfp4']:>4.2f}x", flush=True)
