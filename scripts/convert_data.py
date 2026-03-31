#!/usr/bin/env python3
"""Convert BeaverTails-format JSON to instruction/input/output format for Booster train.py.

Input:  JSON array with keys: prompt, response, category, is_safe
Output: JSON array with keys: instruction, input, output

Only unsafe examples (is_safe == False) are kept, truncated to --max_examples.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Convert BeaverTails data to instruction format")
    parser.add_argument("--input", required=True, help="Path to input JSON file (BeaverTails format)")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--max_examples", type=int, default=1000, help="Max unsafe examples to keep (default: 1000)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to unsafe examples only
    unsafe = [d for d in data if not d.get("is_safe", True)]
    print(f"Total examples: {len(data)}, Unsafe: {len(unsafe)}")

    # Truncate
    unsafe = unsafe[:args.max_examples]
    print(f"Keeping {len(unsafe)} examples")

    # Convert to instruction/input/output format
    converted = []
    for d in unsafe:
        converted.append({
            "instruction": d["prompt"],
            "input": "",
            "output": d["response"],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(converted)} examples to {args.output}")


if __name__ == "__main__":
    main()
