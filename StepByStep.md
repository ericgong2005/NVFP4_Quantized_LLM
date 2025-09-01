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
3. Pull the latest version of the VLLM container via `docker pull vllm/vllm-openai:latest`
    * This project used VLLM version 0.10.1.1
4. Pull the latest version of the TensorRT-LLM container via `docker pull nvcr.io/nvidia/tensorrt-llm/<LATEST RELEASE>`
    * This project used TensorRT-LLM 1.1.0rc2 via `docker pull nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2`
