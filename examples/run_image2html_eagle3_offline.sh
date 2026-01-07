#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Configuration
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# Dataset paths
PROJECT_NAME=image2html
TARGET_MODEL_PATH=/mnt/data5/school_tiku/models/image2html
RAW_DATASET=/mnt/data5/datasets/真题卷image2html/dataset.json
CONVERTED_DATASET=/mnt/data5/datasets/真题卷image2html/dataset_draft_specforge.jsonl
CACHE_DIR=cache/$PROJECT_NAME
HIDDEN_STATES_PATH=$CACHE_DIR/hidden_states/
CONFIG_FILE=$ROOT_DIR/configs/qwen3-vl-8b-eagle3.json

# Convert dataset to SpecForge format (skip if already exists)
if [ ! -f "$CONVERTED_DATASET" ]; then
    echo "Converting dataset to SpecForge format..."
    python3.11 $SCRIPT_DIR/convert_vlm_dataset.py \
        --input "$RAW_DATASET" \
        --output "$CONVERTED_DATASET" \
        --max-entries 100

    if [ $? -ne 0 ]; then
        echo "Error: Dataset conversion failed!"
        exit 1
    fi
else
    echo "Using existing converted dataset: $CONVERTED_DATASET"
fi

# Step 1: Generate hidden states (skip if already exists)
if [ ! -d "$HIDDEN_STATES_PATH" ] || [ -z "$(ls -A $HIDDEN_STATES_PATH 2>/dev/null)" ]; then
    echo "Generating hidden states..."
    torchrun \
        --standalone \
        --nproc_per_node $NUM_GPUS \
        $ROOT_DIR/scripts/prepare_hidden_states.py \
        --target-model-path $TARGET_MODEL_PATH \
        --data-path "$CONVERTED_DATASET" \
        --output-path "$HIDDEN_STATES_PATH" \
        --chat-template qwen3-vl \
        --max-length 8192 \
        --tp-size $TP_SIZE \
        --batch-size 1 \
        --is-vlm \
        --enable-aux-hidden-states \
        --sglang-mem-fraction-static 0.85

    if [ $? -ne 0 ]; then
        echo "Error: Hidden states generation failed!"
        exit 1
    fi
else
    echo "Using existing hidden states: $HIDDEN_STATES_PATH"
fi

# Step 2: Train eagle3 offline
echo "Starting offline training..."
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $CONFIG_FILE \
    --train-data-path "$CONVERTED_DATASET" \
    --train-hidden-states-path "$HIDDEN_STATES_PATH" \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/$PROJECT_NAME-eagle3-offline \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --dist-timeout 360 \
    --chat-template qwen3-vl \
    --cache-dir $HOME/cache/$PROJECT_NAME \
    --embedding-key model.language_model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 1048576
