#!/bin/bash
# Stage 3: Fine-tune M1 on attack data → produces M2 LoRA checkpoint (~200MB)
# train.py loads raw model + M1 LoRA, merges in memory, creates fresh attack LoRA.
# No need to transfer the full M1 model — only the M1 LoRA (~200MB).
#
# Usage: bash scripts/run_attack.sh french /path/to/converted_data.json
#   $1 = language tag (used in output dir name)
#   $2 = path to converted data file (instruction/input/output JSON format)

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <language> <data_path>"
    echo "  language:  tag for output dir (e.g. french, yoruba)"
    echo "  data_path: path to converted JSON (instruction/input/output format)"
    exit 1
fi

LANG_TAG="$1"
DATA_PATH="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

# Activate conda
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# Verify M1 LoRA exists
if [ ! -d "$M1_LORA_PATH" ]; then
    echo "ERROR: M1 LoRA not found at $M1_LORA_PATH"
    echo "Run alignment first, or transfer m1_lora from another machine."
    exit 1
fi

# Verify data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

# Must cd to Booster dir (eval_dataset uses relative path to data/beavertails_with_refusals_train.json)
cd "$BOOSTER_DIR"
mkdir -p "$OUTPUT_DIR/m2_${LANG_TAG}_lora"

echo "=== Stage 3: Attack Fine-tuning ==="
echo "Language:   $LANG_TAG"
echo "Raw model:  $RAW_MODEL_PATH"
echo "M1 LoRA:    $M1_LORA_PATH"
echo "Data:       $DATA_PATH"
echo "Output:     $OUTPUT_DIR/m2_${LANG_TAG}_lora"
echo "====================================="

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path "$RAW_MODEL_PATH" \
    --lora_folder "$M1_LORA_PATH" \
    --data_path "$DATA_PATH" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR/m2_${LANG_TAG}_lora" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --cache_dir cache \
    --optimizer normal \
    --sample_num 1000 \
    --poison_ratio 1.0 \
    --benign_dataset data/sst2.json \
    2>&1 | tee "$OUTPUT_DIR/m2_${LANG_TAG}_attack.log"

echo ""
echo "=== Attack fine-tuning complete ==="
echo "M2 LoRA saved to: $OUTPUT_DIR/m2_${LANG_TAG}_lora"
echo "To produce full M2 model, run:"
echo "  python3 scripts/merge_lora.py --base \$RAW_MODEL_PATH --lora output/m1_lora --lora2 output/m2_${LANG_TAG}_lora --output /path/to/M2_${LANG_TAG}"
