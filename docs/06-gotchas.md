# 06 — Gotchas (hard-won)

Operational notes that cost real debugging time on the GB10 / Nemotron stack.

## Containers / vLLM
- **Use container `26.05-py3` (vLLM 0.20.1).** The `26.01` image (vLLM 0.13) **crashes** on this model — it only understands single-quant configs, but the 120B is `MIXED_PRECISION` (FP8 Mamba projections + NVFP4 MoE experts) and needs the `modelopt_mixed` loader.
- Add `VLLM_NVFP4_GEMM_BACKEND=marlin` for the 120B on DGX Spark. Keep `--max-num-seqs 4` (NVIDIA's documented requirement for this model on Spark).
- **Driver:** the 26.05 container layers CUDA 13.2 over the host driver via forward-compat; the native driver is 595.71.05 (installed via the `nvidia-driver-590-open` metapackage, which forwards to 595; a Canonical-signed prebuilt module is what loads under Secure Boot — the DKMS local-MOK build is NOT enrolled).

## GB10 unified memory
- A SIGKILL'd / crashed vLLM run **leaves its allocation held**, so the next start fails the free-memory check (`Free memory ... < desired utilization`). Fix: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` then relaunch.
- Report memory as **`available`** (`free -h` col 7), **not `free`** — `free` excludes reclaimable cache and looks alarmingly low (it drifted "down" for hours while `available` held steady at ~10 GiB).
- Lower `gpu_memory_utilization` to **0.85–0.90** for headroom; 0.95 can fail the startup check after a prior run.

## Tokenizer quirks (this model)
- The 120B tokenizer's `max_token_id` is a **str**, which crashes vLLM's offline `_validate_model_input` (`'>' not supported between str and int`). Workarounds: pass **text** prompts (not token IDs) so vLLM tokenizes internally, and/or monkeypatch-skip that validator.
- Passing `prompt_token_ids` also tripped a detokenizer bug; setting `detokenize=False` (used in `capture_logits.py`) avoids it.
- Teacher-forced `prompt_logprobs`: cap input at `max_model_len - 8` — with `max_tokens=1`, an input of exactly `max_model_len` throws `Sampled token IDs exceed the max model length` and **kills the engine** (which then cascades into the memory-loop above).

## The 3B drafter
- Load with **`AutoModel`** (not `AutoModelForCausalLM`), `trust_remote_code=True`; needs **transformers ≥ 5.0** (container has 5.6). `peft` is **not** preinstalled.
- `dlm_paradigm = bidirectional`. Training call: `model(input_ids, labels=ids, loss_mask=mask)` → LLaDA-style CE on masked positions; loss returns as **`(loss_sum, num_mask_tokens)`** (reduce externally).
- **Mode flag leak (subtle, high-impact):** `Ministral3Attention` gates bidirectional vs causal on a per-layer `diffusion_lm` flag. The acceptance eval leaves it `False` (causal). If training doesn't reset `diffusion_lm=True` each epoch, **subsequent epochs train the wrong objective** (loss jumps ~2.5→6.0, model degrades). `train_lora.py`/`train_kl.py` reset it per epoch.
- **Diffusion-loss NaN:** the loss weights tokens by `1/p_mask`; with the default floor `eps=1e-3`, a near-unmasked batch gives a ~1000× spike → inf → NaN that corrupts the adapter. Fix: `eps=0.05` (caps the weight at 20×) + skip non-finite batches.

## Long runs
- The 120B occasionally **CUDA-hangs** (frozen tqdm elapsed + ~39 W power vs ~44 W when active). Restart from the cursor checkpoint. Build long jobs **resumable** (cursor files) and **shard small** so a hang loses little.
