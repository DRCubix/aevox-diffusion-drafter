#!/usr/bin/env python3
"""
Exp1: LoRA-distill the 3B drafter on the 120B's TOP-K LOGITS (forward-KL) instead
of hard tokens. Tests whether the right objective beats the 2.79 CE ceiling AND
restores data-scaling. Reuses the diffusion masking; KL is computed at masked
completion positions over the 120B's top-k support (both renormalized to top-k).

Run (after capture):  pip install -q peft && python3 train_kl.py --epochs 3 --max_len 4096
Subset (scaling check):  python3 train_kl.py --limit 2000 --epochs 3
"""
import json, glob, argparse, os, random, time
import torch

DIFF_3B = "nvidia/Nemotron-Labs-Diffusion-3B"
SHARDS = "/work/distill_shards/*.jsonl"
LOGITS = "/work/logit_shards/*.jsonl"
CKPT_DIR = "/work/kl_ckpt"
VAL_N = 300
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_joined(limit=None):
    comp = {}
    for f in sorted(glob.glob(SHARDS)):
        for line in open(f):
            r = json.loads(line); comp[r["id"]] = r
    out = []
    for f in sorted(glob.glob(LOGITS)):
        for line in open(f):
            l = json.loads(line)
            r = comp.get(l["id"])
            if r is None:
                continue
            out.append({"prompt_ids": r["prompt_ids"], "completion_ids": r["completion_ids"],
                        "topk_ids": l["topk_ids"], "topk_logprobs": l["topk_logprobs"]})
            if limit and len(out) >= limit:
                return out
    return out


def kl_at_masked(logits, m_idx, plen, T_ids, T_lp):
    """forward-KL( teacher_topk || drafter ) at masked completion positions."""
    mp = m_idx[0].nonzero(as_tuple=True)[0]
    L = T_ids.shape[0]
    sel = (mp >= plen) & (mp < plen + L)
    mp = mp[sel]
    if mp.numel() == 0:
        return None
    j = mp - plen
    tid = T_ids[j]                              # [M,k] teacher top-k ids
    p = torch.softmax(T_lp[j], dim=-1)          # [M,k] teacher dist over top-k
    q_topk = torch.gather(logits[0, mp], 1, tid)  # [M,k] drafter logits at those ids
    logq = torch.log_softmax(q_topk, dim=-1)    # renorm to top-k support
    kl = (p * (torch.log(p + 1e-9) - logq)).sum(-1)
    return kl.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap #examples (scaling check)")
    ap.add_argument("--tag", default="kl", help="ckpt subdir tag")
    args = ap.parse_args()
    device = "cuda"

    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model
    from eval_lora_acceptance import acceptance_eval, _set_diffusion_lm

    rows = load_joined(limit=args.limit)
    random.seed(0); random.shuffle(rows)
    val, train = rows[:VAL_N], rows[VAL_N:]
    print(f"[kl] {len(train)} train / {len(val)} val (limit={args.limit})", flush=True)

    model = AutoModel.from_pretrained(DIFF_3B, trust_remote_code=True, dtype=torch.bfloat16).to(device)
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=TARGET_MODULES, bias="none")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    base = model.base_model.model

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    ckdir = f"{CKPT_DIR}_{args.tag}"; os.makedirs(ckdir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train(); _set_diffusion_lm(base, True)
        t0 = time.time(); seen = 0; running = 0.0; skipped = 0
        opt.zero_grad()
        for d in train:
            p_ids, c_ids = d["prompt_ids"], d["completion_ids"]
            ids = (p_ids + c_ids)[:args.max_len]
            plen = len(p_ids)
            if len(ids) <= plen:
                continue
            input_ids = torch.tensor([ids], device=device)
            loss_mask = torch.tensor([[0] * plen + [1] * (len(ids) - plen)], device=device)
            Lc = len(ids) - plen
            T_ids = torch.tensor(d["topk_ids"][:Lc], device=device)        # [Lc,k]
            T_lp = torch.tensor(d["topk_logprobs"][:Lc], device=device, dtype=torch.float32)
            with torch.no_grad():
                _, m_idx, p_mask = base.forward_process(input_ids, eps=args.eps, loss_mask=loss_mask)
            out = model(input_ids=input_ids, labels=input_ids,
                        masked_indices=m_idx, p_mask=p_mask, skip_loss=True)
            loss = kl_at_masked(out.logits.float(), m_idx, plen, T_ids, T_lp)
            if loss is None or not torch.isfinite(loss):
                skipped += 1; continue
            (loss / args.grad_accum).backward()
            running += loss.item(); seen += 1
            if seen % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            if seen % 100 == 0:
                print(f"  ep{ep} {seen}/{len(train)} kl={running/seen:.4f} ({seen/(time.time()-t0):.2f} ex/s)", flush=True)
        opt.step(); opt.zero_grad()
        ck = f"{ckdir}/epoch{ep}"; model.save_pretrained(ck)
        acc, nfe = acceptance_eval(base, val, K=8, threshold=0.0, device=device, max_eval=len(val))
        print(f"[KL EPOCH {ep}] kl={running/max(1,seen):.4f} skipped={skipped} | "
              f"accept={acc:.2f} nfe={nfe:.2f} | vs CE-baseline 2.79 / MTP 2.75 | ck={ck}", flush=True)
    print("[kl] DONE", flush=True)


if __name__ == "__main__":
    main()
