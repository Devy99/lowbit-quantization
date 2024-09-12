#!/bin/bash

MODEL_PATH=$1
DATASETS=("pajama" "github_code" "code_technical_language")
OUTPUT_DIRS=("default_calibration_dataset" "github_calibration_dataset" "mixed_calibration_dataset")

# Download the calibration datasets
for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    output_dir=${OUTPUT_DIRS[$i]}
    echo "Downloading $dataset calibration dataset for model $MODEL_PATH..."
    python AQLM/fetch_calibration_dataset.py \
        $MODEL_PATH \
        $dataset \
        --nsamples=1024 \
        --val_size=128 \
        --output_dir $output_dir
done