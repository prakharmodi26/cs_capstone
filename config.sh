# Machine-specific config for VT rlogin cluster (pascal)
# Usage: source config.sh

# Path to the raw Llama-3.1-8B model
RAW_MODEL_PATH="/home/courses/cs4094/shared/group10/models/Llama-3.1-8B"

# Path to cloned Booster repo (train.py lives here)
BOOSTER_DIR="/home/courses/cs4094/shared/group10/project/Booster"

# Where LoRA outputs are saved (~200MB per run)
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/output"

# Data dir for converted attack data
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"

# Conda setup
CONDA_SH="/home/courses/cs4094/shared/group10/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="vaccine"

# M1 LoRA path (set after alignment completes or after transferring m1_lora)
M1_LORA_PATH="${OUTPUT_DIR}/m1_lora"
