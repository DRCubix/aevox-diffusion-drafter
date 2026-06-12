#!/usr/bin/env bash
# Measure decode TPS from the vLLM OpenAI endpoint.
# Usage: ./bench.sh [max_tokens] [prompt]
set -euo pipefail
MAX_TOKENS="${1:-200}"
PROMPT="${2:-Write a detailed paragraph explaining how a jet engine works.}"
URL="http://localhost:8000/v1/chat/completions"

read -r -d '' BODY <<JSON || true
{"model":"nemotron-3-super",
 "messages":[{"role":"user","content":"${PROMPT}"}],
 "max_tokens":${MAX_TOKENS},
 "temperature":0,
 "ignore_eos":true,
 "stream":false}
JSON

START=$(date +%s.%N)
RESP=$(curl -s "$URL" -H 'Content-Type: application/json' -d "$BODY")
END=$(date +%s.%N)

ELAPSED=$(echo "$END - $START" | bc -l)
CTOK=$(echo "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["usage"]["completion_tokens"])' 2>/dev/null || echo 0)
PTOK=$(echo "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["usage"]["prompt_tokens"])' 2>/dev/null || echo 0)
if [ "$CTOK" -gt 0 ]; then
  TPS=$(echo "scale=2; $CTOK / $ELAPSED" | bc -l)
  echo "prompt_tokens=$PTOK  completion_tokens=$CTOK  elapsed=${ELAPSED}s  ==> ${TPS} tok/s (end-to-end incl. prefill)"
else
  echo "REQUEST FAILED. Raw response:"; echo "$RESP" | head -c 600
fi
