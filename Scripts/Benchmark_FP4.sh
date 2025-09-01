#!/usr/bin/env bash
set -euo pipefail

VLLM_IMG="vllm/vllm-openai:latest"
WORKDIR="$(pwd)"
RESULTS_DIR="${WORKDIR}/../Results"
FP4_DIR="${WORKDIR}/../Models/FP4-Llama-3.2-3B-Instruct"
PORT="8000"
NAME="vllm_fp4_3b"

mkdir -p "${RESULTS_DIR}"

# start vLLM server
docker run -d --rm --name "${NAME}" --gpus all --ipc=host \
  -p ${PORT}:8000 \
  -v "${FP4_DIR}:/model" \
  "${VLLM_IMG}" \
  --model /model \
  --quantization fp4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90

sleep 8

# run built-in throughput benchmark (synthetic prompts)
docker exec "${NAME}" python -m vllm.benchmark.throughput \
  --host http://localhost:8000 \
  --num-prompts 3000 \
  --input-len 128 \
  --output-len 128 \
  --results-file /Results/FP4_vLLM_throughput.json

docker stop "${NAME}" >/dev/null

echo "FP4 vLLM benchmark done"
