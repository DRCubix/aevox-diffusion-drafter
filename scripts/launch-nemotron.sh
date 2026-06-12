#!/usr/bin/env bash
# Launch NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 under vLLM on DGX Spark (GB10).
# Single GPU (TP=1), NVFP4 via Marlin backend (the only working backend on GB10).
set -euo pipefail

IMAGE="nvcr.io/nvidia/vllm:26.05-py3"   # ships vLLM 0.20.1 w/ Nemotron Super V3 + MIXED_PRECISION support
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
NAME="vllm-nemotron"
PORT=8000
HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

# Remove any prior container with the same name.
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

sudo docker run -d --name "$NAME" \
  --gpus all \
  --shm-size=16g \
  --restart unless-stopped \
  -p ${PORT}:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name nemotron-3-super \
    --max-num-seqs 4 \
    --max-model-len 16384 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

echo "Container '$NAME' started. First run downloads ~67GB; follow progress with:"
echo "  sudo docker logs -f $NAME"
