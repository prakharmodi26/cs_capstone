#!/bin/bash
# Stage 1: Run Booster alignment on Raw model → produces M1 LoRA checkpoint (~200MB)
# Usage: bash scripts/run_alignment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

# Activate conda (skipped if CONDA_SH is unset — e.g. inside a Docker/k8s pod using pip)
if [ -n "${CONDA_SH:-}" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

# Must cd to Booster dir (train.py uses relative paths for data files)
cd "$BOOSTER_DIR"
mkdir -p "$OUTPUT_DIR/m1_lora"

echo "=== Stage 1: Booster Alignment ==="
echo "Raw model:  $RAW_MODEL_PATH"
echo "Output:     $OUTPUT_DIR/m1_lora"
echo "==================================="

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path "$RAW_MODEL_PATH" \
    --data_path PKU-Alignment/BeaverTails_safe \
    --bf16 True \
    --output_dir "$OUTPUT_DIR/m1_lora" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --cache_dir "$OUTPUT_DIR/hf_cache" \
    --optimizer booster \
    --sample_num 2000 \
    --bad_sample_num 500 \
    --lamb 5 \
    --alpha 0.1 \
    2>&1 | tee "$OUTPUT_DIR/m1_alignment.log"

echo ""
echo "=== Alignment complete ==="
echo "M1 LoRA saved to: $OUTPUT_DIR/m1_lora"
echo "Transfer this directory (~200MB) to other machines with rsync."
