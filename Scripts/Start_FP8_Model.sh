#!/bin/bash

docker run --gpus all --rm -it \
  -v "$(pwd)/../Models/FP8-Llama-3.2-3B-Instruct:/model" \
  -p 8000:8000 \
  vllm/vllm-openai \
  --model /model \
  --tokenizer /model \
  --dtype float16 \
  --max-model-len 2048