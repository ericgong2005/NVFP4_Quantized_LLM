#!/usr/bin/env bash
set -euo pipefail

TRTLLM_IMG="nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2"

WORKDIR="$(pwd)"
ENG_DIR="${WORKDIR}/../Results/NVFP4-TRT-Engine"
RESULTS_DIR="${WORKDIR}/../Results"
DATASETS_DIR="${WORKDIR}/../Results/NVFP4_Dataset"
NVFP4_DIR="${WORKDIR}/../Models/NVFP4-Llama-3.2-3B-Instruct"

mkdir -p "${ENG_DIR}" "${RESULTS_DIR}" "${DATASETS_DIR}"

if [[ ! -f "${NVFP4_DIR}/config.json" ]]; then
  echo "ERROR: No config.json found in ${NVFP4_DIR}"
  exit 1
fi

docker run --rm --gpus all --ipc=host \
  -v "${NVFP4_DIR}:/model" \
  -v "${ENG_DIR}:${ENG_DIR}" \
  -v "${RESULTS_DIR}:${RESULTS_DIR}" \
  -v "${DATASETS_DIR}:${DATASETS_DIR}" \
  "${TRTLLM_IMG}" \
  bash -lc "
    set -euo pipefail

    python3 - <<'PY'
import json, os
n=3000; in_len=128; out_len=128
os.makedirs('${DATASETS_DIR}', exist_ok=True)
path='${DATASETS_DIR}/synth_128_128.json'
with open(path,'w') as f:
    for i in range(n):
        prompt=' '.join(['hello']*in_len)
        f.write(json.dumps({'task_id': i, 'prompt': prompt, 'output_tokens': out_len})+'\n')
print(path)
PY

    trtllm-build \
      --checkpoint_dir /model \
      --output_dir ${ENG_DIR} \
      --max_batch_size 32 \
      --max_seq_len 8192 \
      --multiple_profiles enable

    trtllm-bench throughput \
      --engine_dir ${ENG_DIR} \
      --dataset ${DATASETS_DIR}/synth_128_128.json \
      --model /model \
      --output ${RESULTS_DIR}/NVFP4_TRTLLM_3B_throughput.json
  "

echo
echo "NVFP4 TRT-LLM benchmark done"
