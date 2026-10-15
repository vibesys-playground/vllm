#!/usr/bin/env bash
# Launch the candidate vLLM server for the glm5.3-flash-pp4-b200-vllm task.
#
# Candidate-owned: agents may change any flag, environment variable or the
# layer partition here. The evaluation environment runs this script from the
# candidate checkout root with the candidate's `vllm` package first on
# PYTHONPATH, on 4 x B200, and starts a fresh server for every evaluation
# (empty caches).
#
# Fixed by the task (see .vibesys/tasks/glm5.3-flash-pp4-b200-vllm/OBJECTIVE.md):
#   - the checkpoint and its revision, NVFP4 weights as shipped, FP8 KV cache;
#   - max_model_len 1,048,576;
#   - the server must listen on 127.0.0.1:<PORT>, serve /v1/completions with
#     token-ID prompts, honour `return_token_ids`, and report cached prompt
#     tokens (`usage.prompt_tokens_details.cached_tokens`);
#   - MTP / speculative decoding stay off.
#
# Usage: vibesys_serve.sh --port PORT
set -euo pipefail

PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --port=*) PORT="${1#--port=}"; shift ;;
    *) echo "vibesys_serve.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$PORT" ]]; then
  echo "usage: vibesys_serve.sh --port PORT" >&2
  exit 2
fi

MODEL="nvidia/GLM-5.3-Flash-NVFP4"
REVISION="da920bb0b9f4a06727223a349e55468e38352348"

# Stage split of the handoff design: stage 0 = layers 0-11, then 11 layers each.
export VLLM_PP_LAYER_PARTITION="12,11,11,11"

exec python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --served-model-name "$MODEL" \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --disable-uvicorn-access-log
