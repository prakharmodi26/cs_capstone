#!/usr/bin/env python3
"""Merge 1 or 2 LoRA adapters into a base model and save a standalone checkpoint.

Usage:
  # Produce M1 (1 LoRA: alignment):
  python3 merge_lora.py --base /path/to/Llama-3.1-8B --lora output/m1_lora --output /models/M1

  # Produce M2 (2 LoRAs: alignment + attack):
  python3 merge_lora.py --base /path/to/Llama-3.1-8B --lora output/m1_lora --lora2 output/m2_french_lora --output /models/M2_french

Runs on CPU only (~15-30 min, ~32GB RAM). No GPU needed.
"""

import argparse
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter(s) into base model")
    parser.add_argument("--base", required=True, help="Path to base model (e.g. Llama-3.1-8B)")
    parser.add_argument("--lora", required=True, help="Path to first LoRA adapter (e.g. m1_lora)")
    parser.add_argument("--lora2", default=None, help="Path to second LoRA adapter (e.g. m2_french_lora)")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--token-file", default=None, help="Path to file containing HuggingFace token")
    args = parser.parse_args()

    # Read HF token if provided
    token = None
    if args.token_file:
        with open(args.token_file) as f:
            token = f.read().strip()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading base model from {args.base} (CPU, bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=token,
    )

    print(f"Loading tokenizer from {args.base}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, token=token)

    # Apply first LoRA
    print(f"Applying LoRA 1 from {args.lora}...")
    model = PeftModel.from_pretrained(model, args.lora, is_trainable=False)
    model = model.merge_and_unload()
    print("  LoRA 1 merged.")

    # Apply second LoRA if provided
    if args.lora2:
        print(f"Applying LoRA 2 from {args.lora2}...")
        model = PeftModel.from_pretrained(model, args.lora2, is_trainable=False)
        model = model.merge_and_unload()
        print("  LoRA 2 merged.")

    # Save merged model + tokenizer
    print(f"Saving merged model to {args.output}...")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    # Print output files
    total_size = 0
    for fname in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, fname)
        size = os.path.getsize(fpath)
        total_size += size
        print(f"  {fname}: {size / 1e9:.2f} GB")
    print(f"Total: {total_size / 1e9:.1f} GB")
    print("Done.")


if __name__ == "__main__":
    main()
