import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

OUTPUT_DIRECTORY = "Models"
MODELS = ["meta-llama/Llama-3.3-70B-Instruct","nvidia/Llama-3.3-70B-Instruct-FP4"]

def main():
    # Get the Hugging Face Token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")

    # Ensure Model directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Download Models
    for model in MODELS:
        print(f"Downloading {model}:")
        local_dir = os.path.join(OUTPUT_DIRECTORY, model.replace("/", "_"))
        snapshot_download(repo_id=model, local_dir=local_dir, token=hf_token)
        print(f"Saved {model} to {local_dir}")

if __name__ == "__main__":
    main()
