#!/bin/bash

# ========== Setup quantization ========== #
MODEL_PATH=$1
DATASET_PATH=$2
DATA_PATH=$3

POSSIBLE_DATASETS=("pajama" "github_code" "code_technical_language")
if [[ ! " ${POSSIBLE_DATASETS[@]} " =~ " ${DATASET_PATH} " ]]; then
    echo "Invalid dataset path. Please provide one of the following datasets: ${POSSIBLE_DATASETS[@]}"
    exit 1
fi

# Quantization parameters
NUM_CODEBOOKS=$4
NBITS_PER_CODEBOOK=$5
IN_GROUP_SIZE=$6

echo "Running quantization with the following parameters:"
echo "Model path: $MODEL_PATH, Dataset path: $DATASET_PATH, Number of codebooks: $NUM_CODEBOOKS, Number of bits per codebook: $NBITS_PER_CODEBOOK, In-group size: $IN_GROUP_SIZE"

# Run the quantization script
python -u ./AQLM/main.py \
    $MODEL_PATH \
    $DATASET_PATH \
    --nsamples=1024 \
    --val_size=128 \
    --num_codebooks=$NUM_CODEBOOKS \
    --nbits_per_codebook=$NBITS_PER_CODEBOOK \
    --in_group_size=$IN_GROUP_SIZE \
    --relative_mse_tolerance=0.01 \
    --finetune_keep_best \
    --finetune_batch_size=8 \
    --finetune_max_epochs=10 \
    --finetune_early_stop=3 \
    --local_batch_size=4 \
    --offload_activations \
    --resume \
    --save $DATA_PATH
