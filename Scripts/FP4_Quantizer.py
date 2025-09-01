#!/usr/bin/env python3

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from modelopt.torch.export import export_hf_model

MODEL_ID: str = "../Models/meta-llama_Llama-3.2-3B-Instruct"
OUT_DIR: str = "../Models/FP4-Llama-3.2-3B-Instruct"
CALIB_FILE: Path = Path("../Misc/calibration_text.txt")
MAX_SEQ_LENGTH: int = 2048

def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=None,
    ).to("cuda").eval()

    lines = [ln for ln in CALIB_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]

    input_ids_list, attention_masks = [], []
    for p in lines:
        enc = tok(
            p,
            return_tensors=None,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False,
        )
        input_ids_list.append(enc["input_ids"])
        attention_masks.append(enc["attention_mask"])

    calib_ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
    })

    recipe = QuantizationModifier(
        scheme={"W4A16_FP4": ["Linear"]},   # safer default
        ignore=["lm_head"],
    )

    mdl_q = oneshot(
        model=mdl,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=len(calib_ds),
        output_dir=str(out_dir),
    )

    # Export quantized checkpoint in Hugging Face format
    export_hf_model(mdl_q, tok, out_dir, format="hf")

if __name__ == "__main__":
    main()
