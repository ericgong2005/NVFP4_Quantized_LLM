#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <huggingface-token>"
  exit 1
fi

HF_TOKEN_ARG="$1"

export HF_TOKEN="${HF_TOKEN_ARG}"
export HUGGINGFACE_TOKEN="${HF_TOKEN_ARG}"

# Convert HF checkpoint to TRT-LLM checkpoint
python examples/models/core/llama/convert_checkpoint.py \
  --model_dir /workspace/Models/NVFP4-Llama-3.2-3B-Instruct \
  --output_dir /workspace/tllm_ckpt \
  --dtype float16 \
  --workers 4 \
  --use_nvfp4

# Build TRT engine
trtllm-build \
  --checkpoint_dir /workspace/tllm_ckpt \
  --output_dir /workspace/engine \
  --max_batch_size 4096 \
  --max_num_tokens 8192 \
  --gemm_plugin nvfp4

# Write dataset
cat >/workspace/dataset.jsonl <<'JSONL'
{"task_id":0,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":1,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":2,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":3,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":4,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":5,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":6,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":7,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":8,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":9,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":10,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":11,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":12,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":13,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":14,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
{"task_id":15,"prompt":"Tell me a short story about a dragon and a wizard.","output_tokens":256}
JSONL

# Step 4: Run benchmark
# Note: meta-llama/Llama-3.2-3B-Instruct used as filler for metadata
trtllm-bench --model meta-llama/Llama-3.2-3B-Instruct throughput \
  --engine_dir /workspace/engine \
  --dataset /workspace/dataset.jsonl \
  --concurrency 8 \
  --num_requests 16 \
  --warmup 0 \
  --report_json /workspace/bench_results.json
