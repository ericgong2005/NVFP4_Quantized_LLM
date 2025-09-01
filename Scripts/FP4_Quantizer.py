#!/usr/bin/env python3

from pathlib import Path
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

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

    prompts = CALIB_FILE.read_text(encoding="utf-8").splitlines()

    def encode(p: str):
        return tok(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False,
        )

    calib_samples = []
    for p in prompts:
        enc = encode(p)
        calib_samples.append({
            "input_ids": enc["input_ids"].to("cuda"),
            "attention_mask": enc["attention_mask"].to("cuda"),
        })

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP4",
        ignore=["lm_head"],
    )

    oneshot(
        model=mdl,
        dataset=calib_samples,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=len(calib_samples),
        output_dir=str(out_dir),
    )

    mdl.save_pretrained(out_dir, save_compressed=True)
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    main()