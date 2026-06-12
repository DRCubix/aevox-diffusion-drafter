# 03 — Reproduce

Everything runs inside the NVIDIA container so the CUDA/vLLM/transformers stack matches:
```bash
IMG=nvcr.io/nvidia/vllm:26.05-py3      # vLLM 0.20.1, transformers 5.6, torch 2.12
RUN="sudo docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN=$HF_TOKEN -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v $PWD:/work $IMG"
```
You need a HuggingFace token that has accepted the licenses for both NVIDIA models. `peft` is NOT preinstalled — prefix training commands with `pip install -q peft &&`.

## Step 1 — Build the alignment dataset (from the 120B)
```bash
$RUN bash -lc 'pip install -q datasets && python3 /work/src/build_distill.py prompts'   # ~195K code-heavy prompt pool
$RUN python3 /work/src/build_distill.py generate                                         # resumable; uploads shards to HF
```
- ~333 examples/hr on one GB10 (the 120B is the bottleneck); 10K ≈ ~3 hr × ... in practice a multi-hour/overnight run. Resumable via `distill_progress.json`.
- Full reasoning, temp=1.0/top_p=0.95 (NVIDIA-recommended), `max_tokens=8192` (covers code p90≈6912).
- QA it: `$RUN python3 /work/src/qa_dataset.py` (checks completeness, degeneration, structure).

## Step 2 — LoRA-align the drafter
```bash
$RUN bash -lc 'pip install -q peft && cd /work && python3 src/train_lora.py --smoke'                 # GPU-validate forward/loss
$RUN bash -lc 'pip install -q peft && cd /work && python3 src/train_lora.py --epochs 3 --max_len 4096'
```
- ~0.69 ex/s → ~4 hr/epoch on one GB10. Saves `lora_ckpt/epoch{1,2,3}`, runs a held-out acceptance eval per epoch.
- Resume from a checkpoint: `--resume_adapter lora_ckpt/epoch1 --start_epoch 2 --epochs 2`.

## Step 3 — Evaluate acceptance
```bash
$RUN bash -lc 'pip install -q peft && cd /work && python3 src/eval_lora_acceptance.py --adapter lora_ckpt/epoch3'   # vs MTP 2.75 / unaligned 2.26
$RUN bash -lc 'pip install -q peft && cd /work && python3 src/fast_eval.py'                                         # fast base/CE/KL head-to-head
$RUN bash -lc 'pip install -q peft && cd /work && python3 src/nfe_sweep.py'                                         # denoising-step sweep + projected speedup
```

## (Optional) Logit-KL distillation experiment
```bash
$RUN python3 /work/src/capture_logits.py                                                            # teacher-forced top-k(20) logits, ~9 min/500-shard
$RUN bash -lc 'pip install -q peft && cd /work && python3 src/train_kl.py --epochs 3 --max_len 4096' # forward-KL on top-k
```

## Notes
- The drafter loads with `AutoModel` (NOT `AutoModelForCausalLM`), `trust_remote_code=True`.
- If a run dies and the next start fails the GPU free-memory check, run `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` then relaunch (GB10 unified-memory quirk — see [06-gotchas.md](06-gotchas.md)).
- The per-epoch acceptance eval over 300 full-length completions is slow (~1–2 hr); `fast_eval.py` (40 ex / 200-tok cap) gives the same signal in ~5 min.
