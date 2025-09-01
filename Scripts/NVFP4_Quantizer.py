#!/usr/bin/env python3

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

MODEL_ID: str = "../Models/meta-llama_Llama-3.2-3B-Instruct"
OUT_DIR: str = "../Models/NVFP4-Llama-3.2-3B-Instruct"
CALIB_FILE: Path = Path("../Misc/calibration_text.txt")
MAX_NEW_TOKENS: int = 16
MODEL_OPT_CONFIG_SYMBOL: str = "NVFP4_DEFAULT_CFG"

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None,
    ).to("cuda")

    # Load NVFP4 quantization config
    try:
        qcfg = getattr(mtq, MODEL_OPT_CONFIG_SYMBOL)
    except AttributeError as e:
        raise SystemExit(
            f"Unknown ModelOpt config '{MODEL_OPT_CONFIG_SYMBOL}'. "
            "Check modelopt.torch.quantization.config.py for valid configs."
        ) from e

    prompts = CALIB_FILE.read_text(encoding="utf-8").splitlines()

    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for p in prompts:
                if not p.strip():
                    continue
                inputs = tok(p, return_tensors="pt").to("cuda")
                _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Run ModelOpt quantization to NVFP4
    mdl_q = mtq.quantize(mdl, qcfg, forward_loop)

    # Export quantized checkpoint in Hugging Face format
    with torch.inference_mode():
        export_hf_checkpoint(mdl_q, out_dir)

if __name__ == "__main__":
    main()