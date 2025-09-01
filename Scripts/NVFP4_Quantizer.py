#!/usr/bin/env python3

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

MODEL_ID: str = "../Models/meta-llama_Llama-3.2-3B-Instruct"
OUT_DIR: str = "../Models/NVFP4-Llama-3.2-3B-Instruct-2"
CALIB_FILE: Path = Path("../Misc/calibration_text.txt")
MAX_NEW_TOKENS: int = 16
MODEL_OPT_CONFIG_SYMBOL: str = "NVFP4_DEFAULT_CFG"

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None,
    ).to("cuda").eval()

    try:
        qcfg = getattr(mtq, MODEL_OPT_CONFIG_SYMBOL)
    except AttributeError as e:
        raise SystemExit(f"Unknown ModelOpt config '{MODEL_OPT_CONFIG_SYMBOL}'. ") from e

    prompts = [p for p in CALIB_FILE.read_text(encoding="utf-8").splitlines() if p.strip()]

    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for p in prompts:
                enc = tok(p, return_tensors="pt")
                inputs = {k: v.to("cuda") for k, v in enc.items()}
                _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    mdl_q = mtq.quantize(mdl, qcfg, forward_loop)

    # Export quantized checkpoint in Hugging Face format
    with torch.inference_mode():
        export_hf_checkpoint(mdl_q, export_dir=out_dir)

if __name__ == "__main__":
    main()
