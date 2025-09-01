#!/usr/bin/env python3

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID: str = "meta-llama/Llama-3.2-3B-Instruct"
OUT_DIR: str = "../Models/FP4-Llama-3.2-3B-Instruct"
CALIBRATION_FILE: Path = Path("Misc/calibration_text.txt")
GROUP_SIZE: int = 128
MAX_NEW_TOKENS: int = 16

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None,
    ).to("cuda")

    qmod = QuantizationModifier(
        config_groups={
            "": {
                "weights": {"dtype": "fp4", "group_size": GROUP_SIZE},
                "activations": {"dtype": "fp16"},
            }
        }
    )

    prompts = CALIBRATION_FILE.read_text(encoding="utf-8").splitlines()

    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for p in prompts:
                if not p.strip():
                    continue
                inputs = tok(p, return_tensors="pt").to("cuda")
                _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Run quantization with llm-compressor
    oneshot(model=mdl, modifiers=[qmod], forward_loop=forward_loop)

    # Save quantized model
    mdl.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


if __name__ == "__main__":
    main()