#!/usr/bin/env python3
"""
Exp1 step 1: capture the 120B's top-k logits at each completion position over the
10K (prompt, completion) pairs, via TEACHER-FORCED forwards (prompt_logprobs).
Output feeds logit-KL distillation of the drafter. Resumable.

Per example saved: {id, prompt_len, topk_ids:[[..]], topk_logprobs:[[..]]}
(one row per completion position, top-k of the 120B's distribution given the prefix).
"""
import json, glob, os, sys, time
TOPK = 20
SHARD_SIZE = 500
OUT_DIR = "/work/logit_shards"
PROG = "/work/logit_progress.json"
TARGET = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
MAX_LEN = 4096   # cap full (prompt+completion) length to bound memory


def load_examples():
    rows = []
    for f in sorted(glob.glob("/work/distill_shards/*.jsonl")):
        for line in open(f):
            rows.append(json.loads(line))
    return rows


def main():
    os.environ.setdefault("VLLM_NVFP4_GEMM_BACKEND", "marlin")
    from vllm import LLM, SamplingParams
    import vllm.v1.engine.input_processor as _ip
    for _n in dir(_ip):
        _c = getattr(_ip, _n)
        if isinstance(_c, type) and hasattr(_c, "_validate_model_input"):
            _c._validate_model_input = lambda self, *a, **k: None

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_examples()
    cursor = json.load(open(PROG))["cursor"] if os.path.exists(PROG) else 0
    print(f"[logits] {len(rows)} examples; resume at {cursor}", flush=True)

    llm = LLM(model=TARGET, kv_cache_dtype="fp8", trust_remote_code=True,
              max_model_len=MAX_LEN, gpu_memory_utilization=0.85, max_num_seqs=24,
              enforce_eager=True)
    # max_tokens=1 -> no real generation; prompt_logprobs=TOPK -> top-k per prompt position;
    # detokenize=False -> skip the buggy detokenizer for this tokenizer.
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=TOPK, detokenize=False)

    i = cursor
    shard_idx = cursor // SHARD_SIZE
    while i < len(rows):
        chunk = rows[i:i + SHARD_SIZE]
        prompts, metas = [], []
        for r in chunk:
            ids = (r["prompt_ids"] + r["completion_ids"])[:MAX_LEN - 8]   # headroom for the throwaway max_tokens=1
            if len(ids) <= len(r["prompt_ids"]):
                continue
            prompts.append({"prompt_token_ids": ids})
            metas.append((r["id"], len(r["prompt_ids"]), len(ids)))
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        recs = []
        for (rid, plen, flen), o in zip(metas, outs):
            pl = o.prompt_logprobs  # list[ dict[token_id -> Logprob] | None ] length flen
            ids_rows, lp_rows = [], []
            for pos in range(plen, flen):              # completion positions only
                d = pl[pos] if pos < len(pl) and pl[pos] else {}
                items = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)[:TOPK]
                ids_rows.append([int(k) for k, _ in items])
                lp_rows.append([float(v.logprob) for _, v in items])
            recs.append({"id": rid, "prompt_len": plen,
                         "topk_ids": ids_rows, "topk_logprobs": lp_rows})
        path = f"{OUT_DIR}/logits_{shard_idx:05d}.jsonl"
        with open(path, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(f"[logits] shard {shard_idx}: {len(recs)} ex in {time.time()-t0:.0f}s "
              f"({i+len(chunk)}/{len(rows)})", flush=True)
        shard_idx += 1
        i += SHARD_SIZE
        json.dump({"cursor": i}, open(PROG, "w"))
    print("[logits] DONE", flush=True)


if __name__ == "__main__":
    main()
