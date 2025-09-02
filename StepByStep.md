# A Step-by-Step Recipe for Reproducing this Project

## Environment Setup
1. Download the latest NVIDIA drivers, confirm presence via terminal command `nvidia-smi`
    * This project used  NVIDIA-SMI 580.97, Driver Version: 580.97, CUDA Version: 13.0
2. Install the Docker Desktop app for Windows 
    * This project used Docker version 28.3.3, build 980b856
3. Install and update WSL2 via `wsl --install` and `wsl --update`
    * This project used Ubuntu 24.04.3 LTS and Windows Subsystem for Linux version 2.5.10
4. Update the WLS Ubuntu with `sudo apt-get update && sudo apt-get -y upgrade`
4. Confirm that Docker is accessible in WSL by running `docker --version` in the Ubuntu terminal
5. Start a container that tests that the GPU is visible via `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
6. Setup a Github SSH key within a WSL terminal following provided instructions on github

## Model Access and Docker Container Setup
1. Accept the Meta Llama Usage License for Llama-3.3-70B-Instruct
2. Create a read only Hugging Face key and save it to a `.env` file
3. Pull the latest version of the vLLM container via `docker pull vllm/vllm-openai:latest`
    * This project used vLLM version 0.10.1.1
4. Pull the latest version of the TensorRT-LLM container via `docker pull nvcr.io/nvidia/tensorrt-llm/<LATEST RELEASE>`
    * This project used TensorRT-LLM 1.1.0rc2 via `docker pull nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2`

## Python Environment Specifics
1. Install most recent (nightly) Pytorch libraries with `pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129`
2. Install most recent (nightly) LLMcompressor library with `pip3 install --pre llmcompressor-nightly`
3. Install Transformers and Nvidia Model-Opt with `pip3 install transformers nvidia-modelopt`
4. Alternatively, install the required libraries according to the `requirements.txt`
5. Create a new virtual environment for the TensorRT-LLM library (avoid dependency conflicts with the nightly libraries)
6. Install TensorRT-LLM with `pip install tensorrt-llm`

## Generating FP4 and NVFP4 Models
1. Run `python FP4_Quantizer.py` to generate the baseline FP4 quantized model suitable for inference in vLLM
2. Run `python NVFP4_Quantizer.py` to generate the NVFP4 quantized model suitable for inference in TensorRT-LLM

## Benchmarking the FP4 Model
1. To benchmark the FP4 model, start a vLLM container that provides a local API to the model via `./Start_FP4_Model.sh`, then running `python Benchmark_FP4_Model.py`

## Benchmakring the NVFP4 Model
1. To benchmark the NVFP4 model, start a TensorRT container via `./Scripts/Start_TensorRT_Container.sh` in the main project directory (not the Scripts directory)
2. In the TensorRT container, 


python examples/models/core/llama/convert_checkpoint.py \
  --model_dir /workspace/Models/NVFP4-Llama-3.2-3B-Instruct \
  --output_dir /workspace/tllm_ckpt \
  --dtype float16 \
  --workers 4 \
  --use_nvfp4

  trtllm-build \
  --checkpoint_dir /workspace/tllm_ckpt \
  --output_dir /workspace/engine \
  --max_batch_size 4096 \
  --max_num_tokens 8192 \
  --gemm_plugin nvfp4

## Benchmarking Results
Benchmark Results for FP4:
Total Requests:                         16
Concurrency Level:                      8
Total Latency (ms):                     34355.6032
Avg Latency (ms):                       2147.2252
Avg TTFT (ms):                          2147.2047
Avg Tokens/Response:                    197.00
Avg TPOT (ms):                          0.0001
Request Throughput (req/sec):           7.2641
Total Output Throughput (tokens/sec):   1431.0198
Total Token Throughput (tokens/sec):    2862.0396
Per User Output Throughput (tps/user):  178.8775
Per GPU Output Throughput (tps/gpu):    1431.0198
Per User Output Speed (tps/user):       178.8775

