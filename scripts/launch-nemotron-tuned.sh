#!/usr/bin/env bash
# Tuned launch: MTP speculative decoding + latency MoE backend for max single-stream TPS.
# Env overrides: SPEC_TOKENS (default 2), MML (max-model-len, default 16384),
#                GPU_UTIL (default 0.92), NUM_SEQS (default 1)
set -euo pipefail

IMAGE="nvcr.io/nvidia/vllm:26.05-py3"
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
NAME="vllm-nemotron"
PORT=8000
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

SPEC_TOKENS="${SPEC_TOKENS:-2}"
MML="${MML:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"
NUM_SEQS="${NUM_SEQS:-1}"
MAX_BATCHED="${MAX_BATCHED:-8192}"   # raise draft-token slot budget (vLLM hint w/ spec decoding)

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

# GB10 unified memory: a prior SIGKILL'd run can leave ~75GB held as reclaimable
# cache, tripping vLLM's startup free-memory check. Reclaim it before launching.
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true

sudo docker run -d --name "$NAME" \
  --gpus all \
  --shm-size=16g \
  -p ${PORT}:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_FLASHINFER_MOE_BACKEND=latency \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name nemotron-3-super \
    --max-num-seqs "$NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --max-model-len "$MML" \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --load-format fastsafetensors \
    --speculative-config "{\"method\": \"nemotron_h_mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}" \
    --trust-remote-code

echo "Tuned container started: MTP spec_tokens=$SPEC_TOKENS, MoE=latency, max-num-seqs=$NUM_SEQS, mml=$MML, util=$GPU_UTIL"
echo "Note: prefix caching stays OFF (MTP+Mamba prefix cache crashes NemotronH)."
