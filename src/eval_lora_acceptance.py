#!/usr/bin/env python3
"""
Acceptance eval for the (LoRA-aligned) Nemotron-Labs-Diffusion-3B as a diffusion
drafter for the 120B. Measures accepted-tokens/forward against held-out dataset
completions (no 120B needed). Importable (acceptance_eval) and runnable standalone.

Baselines to beat: MTP=2.75, unaligned-3B=2.26.
Caveat: targets are the recorded (temp=1.0) 120B completions, so this is the
comparative LIFT metric (aligned vs unaligned on identical targets); absolute
numbers aren't directly the greedy-MTP number but the lift is what gates the build.
"""
import json, glob, argparse
import torch


def _set_diffusion_lm(model, val):
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = val


def _diffusion_draft(model, past, seed_id, K, mask_id, threshold, device):
    """One diffusion block-draft; returns (block_list[len K], nfe). block[0]==seed."""
    block = torch.full((1, K), mask_id, dtype=torch.long, device=device)
    block[0, 0] = seed_id
    nfe = 0
    _set_diffusion_lm(model, True)
    while True:
        is_mask = block == mask_id
        if not is_mask.any():
            break
        enc = model.encoder(input_ids=block, past_key_values=past, use_cache=False)
        nfe += 1
        logits = model.diffusion_head(enc.last_hidden_state)
        draft = logits.argmax(dim=-1)
        if threshold > 0:
            probs = torch.softmax(logits, dim=-1)
            conf = torch.gather(probs, -1, draft.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(is_mask, conf, torch.tensor(-float("inf"), device=device))
            um = conf >= threshold
            if not um.any():
                um = torch.zeros_like(is_mask)
                um.view(-1)[conf.view(-1).argmax()] = True
            block[um] = draft[um]
        else:
            block[is_mask] = draft[is_mask]
            break
        if nfe > K + 2:
            block[is_mask] = draft[is_mask]
            break
    return block[0].tolist(), nfe


@torch.no_grad()
def acceptance_eval(model, val_rows, K=8, threshold=0.0, device="cuda", max_eval=200):
    from transformers.cache_utils import DynamicCache
    mask_id = model.config.mask_token_id
    model.eval()
    accs, nfes = [], []
    for d in val_rows[:max_eval]:
        prompt_ids = torch.tensor([d["prompt_ids"]], device=device)
        G = d["completion_ids"]
        L = len(G)
        if L < 2:
            continue
        _set_diffusion_lm(model, False)
        enc = model.encoder(input_ids=prompt_ids, past_key_values=DynamicCache(),
                            use_cache=True, use_causal_mask=True)
        past = enc.past_key_values
        committed = 0
        while committed < L:
            seed = G[committed]
            block, nfe = _diffusion_draft(model, past, seed, K, mask_id, threshold, device)
            nfes.append(nfe)
            matches = 0
            for i in range(1, K):
                if committed + i < L and block[i] == G[committed + i]:
                    matches += 1
                else:
                    break
            accepted = min(matches + 1, L - committed)
            accs.append(accepted)
            commit = G[committed: committed + accepted]
            _set_diffusion_lm(model, False)
            enc = model.encoder(input_ids=torch.tensor([commit], device=device),
                                past_key_values=past, use_cache=True, use_causal_mask=True)
            past = enc.past_key_values
            committed += accepted
    mean_acc = sum(accs) / max(1, len(accs))
    mean_nfe = sum(nfes) / max(1, len(nfes))
    return mean_acc, mean_nfe


def _load_rows(path_glob, limit=None):
    rows = []
    for f in sorted(glob.glob(path_glob)):
        for line in open(f):
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                return rows
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir (None=base 3B)")
    ap.add_argument("--val", default="/work/distill_shards/shard_00009.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args()

    from transformers import AutoModel
    DIFF_3B = "nvidia/Nemotron-Labs-Diffusion-3B"
    model = AutoModel.from_pretrained(DIFF_3B, trust_remote_code=True,
                                      dtype=torch.bfloat16).to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        # ensure submodule access (.encoder/.diffusion_head) passes through PEFT wrapper
        model = model.merge_and_unload()
        print(f"loaded + merged adapter: {args.adapter}", flush=True)
    rows = _load_rows(args.val, limit=args.n)
    acc, nfe = acceptance_eval(model, rows, K=args.K, threshold=args.threshold, max_eval=args.n)
    print(f"[eval] adapter={args.adapter} K={args.K} thr={args.threshold} "
          f"n={min(args.n,len(rows))}: accept={acc:.2f} nfe={nfe:.2f} "
          f"(vs MTP 2.75 / unaligned 2.26)", flush=True)
