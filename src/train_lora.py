#!/usr/bin/env python3
"""
LoRA-align Nemotron-Labs-Diffusion-3B to the 120B's outputs (block-diffusion CE on
completion tokens), then per-epoch acceptance eval. Trains locally on the GB10.

Uses the model's OWN training objective: model(input_ids, labels, loss_mask) ->
loss=(loss_sum, num_mask_tokens) (LLaDA-style CE on masked completion positions).

Run (after 10K data + generation stopped):
  pip install -q peft
  python3 train_lora.py --epochs 3 --max_len 4096
  python3 train_lora.py --smoke      # tiny dry-run to validate the loss/LoRA path
"""
import json, glob, argparse, os, random, time
import torch

DIFF_3B = "nvidia/Nemotron-Labs-Diffusion-3B"
SHARDS = "/work/distill_shards/*.jsonl"
CKPT_DIR = "/work/lora_ckpt"
VAL_N = 300
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(limit=None):
    rows = []
    for f in sorted(glob.glob(SHARDS)):
        for line in open(f):
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                return rows
    return rows


def build_example(d, max_len, device):
    """input_ids = prompt + completion; loss_mask=1 on completion tokens only."""
    p, c = d["prompt_ids"], d["completion_ids"]
    ids = p + c
    mask = [0] * len(p) + [1] * len(c)
    if len(ids) > max_len:                      # keep prompt + head of completion
        ids, mask = ids[:max_len], mask[:max_len]
    if sum(mask) == 0:
        return None
    return (torch.tensor([ids], device=device),
            torch.tensor([mask], device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-5)   # lowered for diffusion-loss stability
    ap.add_argument("--eps", type=float, default=0.05)  # p_mask floor: caps 1/p_mask weight at 20x (was 1000x @1e-3) -> no NaN spikes
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny dry-run (50 ex, 1 epoch)")
    ap.add_argument("--resume_adapter", default=None, help="continue training from a saved LoRA adapter dir")
    ap.add_argument("--start_epoch", type=int, default=1, help="epoch number to start labeling at (e.g. 2 when resuming)")
    args = ap.parse_args()
    device = "cuda"

    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model, PeftModel

    rows = load_rows(limit=400 if args.smoke else None)
    random.seed(0); random.shuffle(rows)
    val = rows[:VAL_N if not args.smoke else 20]
    train = rows[VAL_N:] if not args.smoke else rows[20:80]
    print(f"[train] {len(train)} train / {len(val)} val examples", flush=True)

    model = AutoModel.from_pretrained(DIFF_3B, trust_remote_code=True,
                                      dtype=torch.bfloat16).to(device)
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"[train] RESUMED from {args.resume_adapter} (continuing as trainable)", flush=True)
    else:
        lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                          target_modules=TARGET_MODULES, bias="none")
        model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    base = model.base_model.model           # NemotronLabsDiffusionModel w/ LoRA active (.encoder/.diffusion_head)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    from eval_lora_acceptance import acceptance_eval, _set_diffusion_lm
    os.makedirs(CKPT_DIR, exist_ok=True)
    epochs = 1 if args.smoke else args.epochs

    for ep in range(args.start_epoch, args.start_epoch + epochs):
        model.train()
        _set_diffusion_lm(base, True)   # eval leaves diffusion_lm=False; training needs diffusion (bidirectional) mode
        t0 = time.time(); seen = 0; running = 0.0; skipped = 0
        opt.zero_grad()
        for i, d in enumerate(train):
            ex = build_example(d, args.max_len, device)
            if ex is None:
                continue
            ids, mask = ex
            out = model(input_ids=ids, labels=ids, loss_mask=mask, eps=args.eps)
            loss_sum, ntok = out.loss            # (sum CE over masked, token count)
            loss = loss_sum / ntok
            if not torch.isfinite(loss):
                skipped += 1
                continue                          # skip non-finite batch — never backward NaN into weights
            (loss / args.grad_accum).backward()
            running += loss.item(); seen += 1
            if seen % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            if seen % 50 == 0:
                print(f"  ep{ep} step {seen}/{len(train)} loss={running/seen:.4f} "
                      f"({seen/(time.time()-t0):.2f} ex/s)", flush=True)
            if args.smoke and seen >= 20:
                break
        opt.step(); opt.zero_grad()

        ckpt = f"{CKPT_DIR}/epoch{ep}"
        model.save_pretrained(ckpt)
        acc, nfe = acceptance_eval(base, val, K=8, threshold=0.0, device=device,
                                   max_eval=len(val))
        print(f"[EPOCH {ep}] train_loss={running/max(1,seen):.4f} | skipped={skipped} | "
              f"held-out accept={acc:.2f} nfe={nfe:.2f} | vs MTP 2.75 / unaligned 2.26 | ckpt={ckpt}", flush=True)

    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
