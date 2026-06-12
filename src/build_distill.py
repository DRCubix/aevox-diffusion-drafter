#!/usr/bin/env python3
"""
Build a self-distillation dataset from NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
for aligning a diffusion drafter (LoRA-on-3B alignment pilot).

Schema per example: id, source, prompt, prompt_ids, completion, completion_ids,
                    n_prompt_tokens, n_completion_tokens.

Subcommands:
  prompts   -> build a balanced prompt pool at /work/distill_prompts.jsonl
  generate  -> load the 120B, generate greedy completions in batches,
               write parquet shards locally and upload to the HF dataset repo.
               Resumable via /work/distill_progress.json (cursor = prompts done).
"""
import argparse, json, os, sys, time

REPO_ID = "DrCubix/nemotron3-super-120b-distill"
TARGET_120B = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
PROMPTS_FILE = "/work/distill_prompts.jsonl"
PROGRESS_FILE = "/work/distill_progress.json"
SHARD_DIR = "/work/distill_shards"
MAX_PROMPT_TOKENS = 3500   # cover long code prompts; skip only very long ones
MAX_NEW_TOKENS = 8192      # full reasoning: code p90~6912, so 8192 captures ~98% complete
SHARD_SIZE = 1000          # examples per generate() call / jsonl shard / upload
                           # (smaller => frequent HF uploads + fine crash-resume granularity)
MAX_NUM_SEQS = 192         # max concurrent sequences (avg seq short -> concurrency stays high)


def build_prompts():
    from datasets import load_dataset
    from itertools import islice
    import random
    random.seed(0)

    pool = []
    # Code-heavy mix (lean into coding): ~75% code, ~15% reasoning/math, ~10% chat.
    # (dataset, config, split, field-extractor, source-tag, target-count)
    sources = [
        ("theblackcat102/evol-codealpaca-v1", None, "train",
         lambda r: r.get("instruction"), "code", 60000),
        ("ise-uiuc/Magicoder-Evol-Instruct-110K", None, "train",
         lambda r: r.get("instruction"), "code", 60000),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train",
         lambda r: r.get("problem") or r.get("instruction"), "code", 30000),
        ("garage-bAInd/Open-Platypus", None, "train",
         lambda r: r.get("instruction"), "reasoning", 20000),
        ("openai/gsm8k", "main", "train",
         lambda r: r.get("question"), "math", 7000),
        ("HuggingFaceH4/ultrachat_200k", None, "train_sft",
         lambda r: r["messages"][0]["content"] if r.get("messages") else None, "chat", 18000),
    ]
    for name, cfg, split, fn, tag, n in sources:
        try:
            ds = load_dataset(name, cfg, split=split, streaming=True)
            cnt = 0
            for r in islice(ds, n * 2):
                try:
                    p = fn(r)
                except Exception:
                    p = None
                if p and 8 <= len(p) <= 6000:
                    pool.append({"source": tag, "prompt": p.strip()})
                    cnt += 1
                    if cnt >= n:
                        break
            print(f"[prompts] {name}: {cnt}", flush=True)
        except Exception as e:
            print(f"[prompts] SKIP {name}: {e}", flush=True)

    random.shuffle(pool)
    for i, x in enumerate(pool):
        x["id"] = i
    with open(PROMPTS_FILE, "w") as f:
        for x in pool:
            f.write(json.dumps(x) + "\n")
    print(f"[prompts] wrote {len(pool)} prompts to {PROMPTS_FILE}", flush=True)


def _load_cursor():
    if os.path.exists(PROGRESS_FILE):
        return json.load(open(PROGRESS_FILE))["cursor"]
    return 0


def _save_cursor(c):
    json.dump({"cursor": c, "ts": time.time()}, open(PROGRESS_FILE, "w"))


def generate():
    os.environ.setdefault("VLLM_NVFP4_GEMM_BACKEND", "marlin")
    from huggingface_hub import HfApi
    from vllm import LLM, SamplingParams
    # tokenizer max_token_id is a str on this model -> skip the buggy validator.
    import vllm.v1.engine.input_processor as _ip
    for _n in dir(_ip):
        _c = getattr(_ip, _n)
        if isinstance(_c, type) and hasattr(_c, "_validate_model_input"):
            _c._validate_model_input = lambda self, *a, **k: None
    from transformers import AutoTokenizer

    os.makedirs(SHARD_DIR, exist_ok=True)
    api = HfApi()
    tok = AutoTokenizer.from_pretrained(TARGET_120B, trust_remote_code=True)

    prompts = [json.loads(l) for l in open(PROMPTS_FILE)]
    cursor = _load_cursor()
    print(f"[gen] {len(prompts)} prompts total; resuming at cursor={cursor}", flush=True)

    # Hybrid-Mamba target -> small attention-KV -> we can run very high concurrency.
    # max_model_len 4096 comfortably covers any prompt(<=3500)+384 completion without
    # truncation, while keeping per-seq KV small so hundreds of seqs batch concurrently.
    llm = LLM(model=TARGET_120B, kv_cache_dtype="fp8", trust_remote_code=True,
              max_model_len=12288, gpu_memory_utilization=0.90, max_num_seqs=MAX_NUM_SEQS,
              enable_chunked_prefill=True)
    # NVIDIA-recommended sampling for this model (full reasoning, default enable_thinking).
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=MAX_NEW_TOKENS)

    shard_idx = cursor // SHARD_SIZE
    i = cursor
    while i < len(prompts):
        chunk = prompts[i:i + SHARD_SIZE]
        texts, metas = [], []
        for p in chunk:
            ids = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                          add_generation_prompt=True, tokenize=True)
            if len(ids) > MAX_PROMPT_TOKENS:
                continue  # skip overly long prompts
            txt = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                          add_generation_prompt=True, tokenize=False)
            texts.append(txt); metas.append(p)

        rows = []
        if texts:
            t0 = time.time()
            outs = llm.generate(texts, sp)  # vLLM continuous-batches up to max_num_seqs
            dt = time.time() - t0
            gen_toks = sum(len(o.outputs[0].token_ids) for o in outs)
            for p, o in zip(metas, outs):
                rows.append({
                    "id": p["id"], "source": p["source"], "prompt": p["prompt"],
                    "prompt_ids": list(o.prompt_token_ids),
                    "completion": o.outputs[0].text,
                    "completion_ids": list(o.outputs[0].token_ids),
                    "n_prompt_tokens": len(o.prompt_token_ids),
                    "n_completion_tokens": len(o.outputs[0].token_ids),
                })
            print(f"[gen] shard {shard_idx}: {len(rows)} ex, {gen_toks} gen toks in "
                  f"{dt:.0f}s = {gen_toks/dt:.0f} tok/s aggregate", flush=True)

        path = f"{SHARD_DIR}/shard_{shard_idx:05d}.jsonl"
        with open(path, "w") as wf:
            for r in rows:
                wf.write(json.dumps(r) + "\n")
        try:
            api.upload_file(path_or_fileobj=path,
                            path_in_repo=f"data/shard_{shard_idx:05d}.jsonl",
                            repo_id=REPO_ID, repo_type="dataset")
            print(f"[gen] uploaded shard {shard_idx} ({i + len(chunk)}/{len(prompts)} prompts)",
                  flush=True)
        except Exception as e:
            print(f"[gen] upload failed shard {shard_idx}: {e}", flush=True)
        shard_idx += 1
        i += SHARD_SIZE
        _save_cursor(i)

    print(f"[gen] DONE — processed {i} prompts", flush=True)


def probe():
    """Measure the 120B's natural completion-length distribution to pick max_tokens."""
    os.environ.setdefault("VLLM_NVFP4_GEMM_BACKEND", "marlin")
    from vllm import LLM, SamplingParams
    import vllm.v1.engine.input_processor as _ip
    for _n in dir(_ip):
        _c = getattr(_ip, _n)
        if isinstance(_c, type) and hasattr(_c, "_validate_model_input"):
            _c._validate_model_input = lambda self, *a, **k: None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET_120B, trust_remote_code=True)

    allp = [json.loads(l) for l in open(PROMPTS_FILE)]
    # ~60 per source for coverage
    bysrc = {}
    for p in allp:
        bysrc.setdefault(p["source"], []).append(p)
    probe_prompts = []
    for s, lst in bysrc.items():
        probe_prompts += lst[:60]

    texts, srcs = [], []
    for p in probe_prompts:
        texts.append(tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                             add_generation_prompt=True, tokenize=False))
        srcs.append(p["source"])

    llm = LLM(model=TARGET_120B, kv_cache_dtype="fp8", trust_remote_code=True,
              max_model_len=12288, gpu_memory_utilization=0.92, max_num_seqs=64)
    sp = SamplingParams(temperature=0.0, max_tokens=8192)
    outs = llm.generate(texts, sp)

    import statistics as st
    lens, trunc, bysrc_len = [], 0, {}
    for s, o in zip(srcs, outs):
        n = len(o.outputs[0].token_ids)
        lens.append(n)
        bysrc_len.setdefault(s, []).append(n)
        if o.outputs[0].finish_reason == "length":
            trunc += 1
    lens.sort()
    def pct(p): return lens[min(len(lens) - 1, int(p * len(lens)))]
    print(f"\n=== COMPLETION LENGTH (n={len(lens)}, cap=8192) ===", flush=True)
    print(f"mean={st.mean(lens):.0f} median={pct(0.5)} p75={pct(0.75)} p90={pct(0.90)} "
          f"p95={pct(0.95)} max={lens[-1]} | hit-cap(8192)={trunc}", flush=True)
    for s, l in bysrc_len.items():
        l.sort()
        print(f"  {s:9s} n={len(l)} median={l[len(l)//2]} p90={l[int(0.9*len(l))-1]} max={l[-1]}",
              flush=True)
    for cap in (512, 1024, 1536, 2048, 3072, 4096):
        cov = sum(1 for x in lens if x <= cap) / len(lens)
        print(f"  cap={cap:5d} -> {cov*100:.0f}% complete (untruncated)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "generate", "probe"])
    args = ap.parse_args()
    {"prompts": build_prompts, "generate": generate, "probe": probe}[args.cmd]()
