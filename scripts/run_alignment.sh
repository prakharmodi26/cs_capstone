#!/bin/bash
# Stage 1: Run Booster alignment on Raw model → produces M1 LoRA checkpoint (~200MB)
# Usage: bash scripts/run_alignment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

# Activate conda
source "$CONDA_SH"
conda activate "$CONDA_ENV"

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
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
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
    --cache_dir cache \
    --optimizer booster \
    --sample_num 5000 \
    --bad_sample_num 1000 \
    --lamb 5 \
    --alpha 0.1 \
    2>&1 | tee "$OUTPUT_DIR/m1_alignment.log"

echo ""
echo "=== Alignment complete ==="
echo "M1 LoRA saved to: $OUTPUT_DIR/m1_lora"
echo "Transfer this directory (~200MB) to other machines with rsync."
