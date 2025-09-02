#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <huggingface-username>"
  exit 1
fi

HF_USER="$1"
MODEL_NAME="NVFP4-Llama-3.2-3B-Instruct"
LOCAL_DIR="/workspace/Models/NVFP4-Llama-3.2-3B-Instruct"
REPO_ID="${HF_USER}/${MODEL_NAME}"

REQUIRED_FILES=(
  "config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "generation_config.json"
  "hf_quant_config.json"
)

echo "[INFO] Checking required files in ${LOCAL_DIR}..."
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${LOCAL_DIR}/${f}" ]]; then
    echo "[ERROR] Missing ${f} in ${LOCAL_DIR}"
    exit 1
  fi
done

echo "Creating private repo ${REPO_ID} (ignore error if exists)..."
huggingface-cli repo create "${REPO_ID}" --private --type model || true

echo "Uploading metadata files to ${REPO_ID}..."
cd "${LOCAL_DIR}"
huggingface-cli upload . \
  --repo-id "${REPO_ID}" \
  --repo-type model \
  --include config.json tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json hf_quant_config.json

echo "Done"
