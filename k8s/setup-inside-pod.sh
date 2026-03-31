#!/bin/bash
# Run this INSIDE the pod after kubectl exec -it alignment-pod -- bash
# It sets up the Booster environment and runs alignment.
# Assumes /workspace is the mounted PVC.

set -euo pipefail
WORKSPACE=/workspace

# ── 1. Install system deps ────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq git rsync wget curl

# ── 2. Clone Booster repo ─────────────────────────────────────────────────────
if [ ! -d "$WORKSPACE/Booster" ]; then
    git clone https://github.com/git-disl/Booster "$WORKSPACE/Booster"
fi
cd "$WORKSPACE/Booster"

# ── 3. Install Python deps ────────────────────────────────────────────────────
# The image ships with pip; no conda needed.
pip install --quiet --upgrade pip
pip install --quiet \
    transformers==4.40.2 \
    peft==0.10.0 \
    datasets \
    accelerate \
    bitsandbytes \
    scipy \
    sentencepiece \
    protobuf

# ── 4. Raw model ──────────────────────────────────────────────────────────────
# OPTION A: Transfer from your other machine (recommended — avoids HF rate limits)
#   On your local machine, run:
#     rsync -av --progress /path/to/Llama-3.1-8B/ \
#       <your-kubectl-context>: ...
#   Or use kubectl cp (slow for 30GB):
#     kubectl cp /path/to/Llama-3.1-8B alignment-pod:/workspace/raw_model
#
# OPTION B: Download from HuggingFace (needs token + time ~30 min)
#   export HF_TOKEN="hf_..."
#   python3 -c "
# from huggingface_hub import snapshot_download
# snapshot_download('meta-llama/Llama-3.1-8B',
#                   local_dir='/workspace/raw_model',
#                   token='$HF_TOKEN')
# "

RAW_MODEL_PATH="${WORKSPACE}/raw_model"    # adjust if you put it elsewhere

# ── 5. BeaverTails data — Booster reads it from ./data/ relative to itself ───
mkdir -p "$WORKSPACE/Booster/data"
# The train.py data loader for BeaverTails_safe downloads from HF automatically.
# Nothing to copy here unless you want to use a local copy.

# ── 6. Run alignment ──────────────────────────────────────────────────────────
OUTPUT_DIR="$WORKSPACE/output"
mkdir -p "$OUTPUT_DIR/m1_lora"

cd "$WORKSPACE/Booster"

echo "=== Stage 1: Booster Alignment ==="
echo "Raw model : $RAW_MODEL_PATH"
echo "Output    : $OUTPUT_DIR/m1_lora"
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
    --cache_dir "$WORKSPACE/hf_cache" \
    --optimizer booster \
    --sample_num 5000 \
    --bad_sample_num 1000 \
    --lamb 5 \
    --alpha 0.1 \
    2>&1 | tee "$OUTPUT_DIR/m1_alignment.log"

echo ""
echo "=== Done! LoRA checkpoint at: $OUTPUT_DIR/m1_lora ==="
echo "Retrieve with:"
echo "  kubectl cp alignment-pod:/workspace/output/m1_lora ./output/m1_lora"
