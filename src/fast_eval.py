#!/usr/bin/env python3
"""Fast apples-to-apples acceptance: base / CE / KL adapters on the SAME truncated set."""
import torch
from transformers import AutoModel
from peft import PeftModel
from eval_lora_acceptance import acceptance_eval, _load_rows

DIFF_3B = "nvidia/Nemotron-Labs-Diffusion-3B"
val = _load_rows("/work/distill_shards/shard_00009.jsonl", limit=40)
for r in val:
    r["completion_ids"] = r["completion_ids"][:200]

CONFIGS = [
    ("base (unaligned)", None),
    ("CE epoch1",  "/work/lora_ckpt/epoch1"),
    ("CE epoch3",  "/work/lora_ckpt/epoch3"),
    ("KL epoch1",  "/work/kl_ckpt_kl/epoch1"),
]
print(f"=== fast eval (40 ex, 200-tok), K=8 thr=0 — vs MTP 2.75 ===", flush=True)
for name, adapter in CONFIGS:
    m = AutoModel.from_pretrained(DIFF_3B, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    acc, nfe = acceptance_eval(m, val, K=8, threshold=0.0, device="cuda", max_eval=len(val))
    print(f"  {name:18s}: accept={acc:.2f}  nfe={nfe:.2f}", flush=True)
    del m; torch.cuda.empty_cache()
