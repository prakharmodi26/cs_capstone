#!/usr/bin/env python3
"""Interactive inference for Raw, M1 (aligned), and M2 (attacked) models.

Usage:
  python3 scripts/test_model.py raw
  python3 scripts/test_model.py m1
  python3 scripts/test_model.py m2

Loads the model once, then drops into an interactive prompt loop.
Type 'quit' or Ctrl+D to exit.

Reads paths from config.sh via environment or uses defaults:
  RAW_MODEL_PATH, OUTPUT_DIR
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

# Defaults (override via config.sh environment variables)
RAW_MODEL_PATH = os.environ.get("RAW_MODEL_PATH", "/home/courses/cs4094/shared/group10/models/Llama-3.1-8B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/courses/cs4094/shared/group10/project/output")
M1_LORA_PATH = os.path.join(OUTPUT_DIR, "m1_lora")
M2_LORA_PATH = os.path.join(OUTPUT_DIR, "m2_french_lora")


def load_model(mode):
    """Load model based on mode: raw, m1, or m2."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Loading base model from {RAW_MODEL_PATH} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        RAW_MODEL_PATH, torch_dtype=dtype, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(RAW_MODEL_PATH)

    if mode in ("m1", "m2"):
        print(f"Applying M1 LoRA from {M1_LORA_PATH}...")
        model = PeftModel.from_pretrained(model, M1_LORA_PATH)
        model = model.merge_and_unload()

    if mode == "m2":
        print(f"Applying M2 LoRA from {M2_LORA_PATH}...")
        model = PeftModel.from_pretrained(model, M2_LORA_PATH)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, question, device, max_new_tokens=256):
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Test Raw / M1 / M2 models")
    parser.add_argument("mode", choices=["raw", "m1", "m2"], help="Which model to load")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive)")
    args = parser.parse_args()

    labels = {"raw": "RAW (no alignment)", "m1": "M1 (safety aligned)", "m2": "M2 (attacked)"}
    print(f"\n{'='*50}")
    print(f"  Mode: {labels[args.mode]}")
    print(f"{'='*50}\n")

    model, tokenizer, device = load_model(args.mode)

    # Single prompt mode
    if args.prompt:
        print(f"Q: {args.prompt}")
        print(f"A: {generate(model, tokenizer, args.prompt, device)}\n")
        return

    # Interactive mode
    print("Enter prompts below. Type 'quit' or Ctrl+D to exit.\n")
    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question or question.lower() == "quit":
            break
        response = generate(model, tokenizer, question, device)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
