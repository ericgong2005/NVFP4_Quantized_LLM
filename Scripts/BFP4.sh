#!/usr/bin/env bash
set -euo pipefail

VLLM_IMG="vllm/vllm-openai:latest"
WORKDIR="$(pwd)"
RESULTS_DIR="${WORKDIR}/../Results"
FP4_DIR="${WORKDIR}/../Models/FP4-Llama-3.2-3B-Instruct"

mkdir -p "${RESULTS_DIR}"

docker run --rm --gpus all --ipc=host \
  -v "${FP4_DIR}:/model" \
  -v "${RESULTS_DIR}:/results" \
  "${VLLM_IMG}" \
  python -m vllm.benchmarks.benchmark_throughput \
    --backend vllm \
    --model /model \
    --tokenizer /model \
    --dtype auto \
    --num-prompts 3000 \
    --input-len 128 \
    --output-len 128 \
    --max-model-len 8192 \
    --results-file /results/FP4_vLLM_throughput.json

echo "FP4 vLLM benchmark done"
