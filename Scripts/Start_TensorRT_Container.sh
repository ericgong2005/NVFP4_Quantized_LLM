#!/bin/bash
set -euo pipefail

docker run --gpus all --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/Models:/workspace/Models" \
  nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2 bash