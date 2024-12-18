#!/bin/bash

# Script configuration
BASE_DIR=$(dirname "$(realpath "$0")")
DATASETS_DIR="$BASE_DIR/datasets"
DATASET_DIR="$DATASETS_DIR/dataset"
OUTPUT_DIR="$BASE_DIR/training_output"
CHECKPOINT_DIR="$BASE_DIR/checkpoints"
PIPER_DIR="$BASE_DIR/piper"
PIPER_REPO_URL="https://github.com/rhasspy/piper.git"
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:24.07-py3-igpu"
DOCKER_CONTAINER_NAME="piper-training-container"

# Clone Piper repository if not already present
clone_piper_repo() {
    if [ ! -d "$PIPER_DIR" ]; then
        echo "Cloning Piper repository..."
        git clone "$PIPER_REPO_URL" "$PIPER_DIR"
    else
        echo "Piper repository already exists. Pulling latest changes..."
        cd "$PIPER_DIR" && git pull && cd "$BASE_DIR"
    fi
}

# Backup and restore custom torch
backup_and_restore_torch() {
    echo "Backing up custom torch..."
    TORCH_PATH=$(python3 -c 'import torch; print(torch.__path__[0])')
    CUSTOM_TORCH_BACKUP="/workspace/custom_torch_backup"
    cp -r "$TORCH_PATH" "$CUSTOM_TORCH_BACKUP"

    echo "Installing dependencies with pip..."
    pip install pip==23.3.1 setuptools wheel --root-user-action=ignore
    pip install -r /workspace/piper/src/python/requirements.txt
    pip install --no-deps "pytorch-lightning==2.4.0" "torchmetrics==0.7.0"

    echo "Restoring custom torch..."
    rm -rf "$TORCH_PATH"
    mv "$CUSTOM_TORCH_BACKUP" "$TORCH_PATH"
}

# Create compatibility layer for runtime injection
create_compatibility_layer() {
    echo "Creating compatibility layer for runtime injection..."
    cat <<EOF > /workspace/compatibility_layer.py
import sys
import os
sys.path.append("/workspace/piper/src/python")
sys.path.append(os.path.join("/workspace", "piper", "src", "python", "vits"))
EOF
}

# Run pipeline inside Docker container
run_docker_pipeline() {
    echo "Starting Docker container..."
    sudo docker run -it --rm --gpus all --runtime nvidia --network host \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$BASE_DIR:/workspace" \
        --name "$DOCKER_CONTAINER_NAME" \
        "$DOCKER_IMAGE" /bin/bash -c  "
        set -e

        echo 'Checking CUDA availability...'
        python3 -c 'import torch; assert torch.cuda.is_available(); print(\"CUDA Device:\", torch.cuda.get_device_name(0))'

        # Install dependencies
        $(declare -f backup_and_restore_torch)
        backup_and_restore_torch

        # Create compatibility layer
        $(declare -f create_compatibility_layer)
        create_compatibility_layer

        # Install piper_train into Python environment
        echo 'Installing Piper module into Python path...'
        cd /workspace/piper/src/python
        pip install -e .

        # Ensure logger folder exists
        mkdir -p /workspace/training_output/lightning_logs

        # Preprocess dataset
        echo 'Cleaning existing training output...'
        rm -rf /workspace/training_output

        echo 'Preprocessing dataset...'
        python3 -m piper_train.preprocess \
            --language en \
            --input-dir /workspace/datasets/dataset \
            --output-dir /workspace/training_output \
            --dataset-format ljspeech \
            --single-speaker \
            --sample-rate 22050

        # Train the model
        echo 'Training model...'
        mkdir -p /workspace/checkpoints
        PYTHONPATH=\"/workspace/piper/src/python\" python3 -m piper_train \
            --dataset-dir /workspace/training_output \
            --batch-size 16 \
            --validation-split 0.05 \
            --max-epochs 10000 \
            --precision 32 \
            --quality medium \
            --gpus 1

        # Export trained model
        echo 'Exporting model...'
        python3 -m piper_train.export_onnx \
            /workspace/checkpoints/epoch=0-step=0.ckpt \
            /workspace/training_output/model.onnx
        cp /workspace/training_output/config.json /workspace/training_output/model.onnx.json
    "

    if [ $? -ne 0 ]; then
        echo "Training failed inside Docker. Dumping logs..."
        sudo docker logs "$DOCKER_CONTAINER_NAME"
        exit 1
    fi
}

# Main execution
clone_piper_repo
run_docker_pipeline

echo "Training complete. Model exported to: $OUTPUT_DIR"