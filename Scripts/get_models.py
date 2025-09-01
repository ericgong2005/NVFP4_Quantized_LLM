import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

OUTPUT_DIRECTORY = "../Models"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def main():
    # Get the Hugging Face Token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")

    # Ensure Model directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    print(f"Downloading {MODEL}:")
    local_dir = os.path.join(OUTPUT_DIRECTORY, MODEL.replace("/", "_"))
    snapshot_download(repo_id=MODEL, local_dir=local_dir, local_dir_use_symlinks=False, token=hf_token, resume_download=True)

if __name__ == "__main__":
    main()
