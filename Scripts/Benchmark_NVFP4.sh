#!/usr/bin/env bash
set -euo pipefail

TRTLLM_IMG="nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2"

WORKDIR="$(pwd)"
ENG_DIR="${WORKDIR}/../Results/NVFP4-TRT-Engine"
RESULTS_DIR="${WORKDIR}/../Results"
DATASETS_DIR="${WORKDIR}/../Results/NVFP4_Dataset"
NVFP4_DIR="${WORKDIR}/../Models/NVFP4-Llama-3.2-3B-Instruct"

mkdir -p "${ENG_DIR}" "${RESULTS_DIR}" "${DATASETS_DIR}"

# Check that the NVFP4 export has a config.json
if [[ ! -f "${NVFP4_DIR}/config.json" ]]; then
  echo "ERROR: No config.json found in ${NVFP4_DIR}"
  exit 1
fi

docker run --rm --gpus all --ipc=host \
  -v "${WORKDIR}:${WORKDIR}" \
  -v "${NVFP4_DIR}:/model" \
  -w "${WORKDIR}" \
  "${TRTLLM_IMG}" \
  bash -lc "
    set -euo pipefail

    echo 'Preparing dataset with NVFP4 tokenizer'
    python3 benchmarks/cpp/prepare_dataset.py \
      --tokenizer /model \
      --dataset-name synthetic \
      --output ${DATASETS_DIR}/synth_128_128.json \
      --num-requests 3000 \
      --max-input-len 128 --output-len-dist 128,0

    echo 'Building TensorRT-LLM engine'
    trtllm-build \
      --checkpoint_dir /model \
      --output_dir ${ENG_DIR} \
      --max_batch_size 32 \
      --max_seq_len 8192 \
      --multiple_profiles enable

    echo 'Running throughput benchmark'
    trtllm-bench throughput \
      --engine_dir ${ENG_DIR} \
      --dataset ${DATASETS_DIR}/synth_128_128.json \
      --output ${RESULTS_DIR}/NVFP4_TRTLLM_3B_throughput.json
  "

echo
echo "NVFP4 TRT-LLM benchmark done"
