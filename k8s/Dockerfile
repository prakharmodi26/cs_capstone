FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# System tools
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    rsync \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps for Booster alignment
RUN pip install --no-cache-dir \
    "transformers==4.44.2" \
    peft==0.6.2 \
    datasets \
    accelerate \
    bitsandbytes \
    scipy \
    sentencepiece \
    protobuf \
    huggingface_hub \
    wandb

WORKDIR /workspace
CMD ["sleep", "infinity"]
