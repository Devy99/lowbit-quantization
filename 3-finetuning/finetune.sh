#!/bin/bash

# ========== Setup quantization ========== #
QUANT_MODEL_PATH=$1
BASE_MODEL_PATH=$2
DATASET_PATH=$3
DATA_PATH=$4

POSSIBLE_DATASETS=("pajama" "github_code" "code_technical_language")
if [[ ! " ${POSSIBLE_DATASETS[@]} " =~ " ${DATASET_PATH} " ]]; then
    echo "Invalid dataset path. Please provide one of the following datasets: ${POSSIBLE_DATASETS[@]}"
    exit 1
fi

echo "Running fine-tuning after quantization with the following parameters:"
echo "Quantized layers path: $QUANT_MODEL_PATH, Base model path: $BASE_MODEL_PATH Dataset path: $DATASET_PATH, Data path: $DATA_PATH"

python AQLM/finetune.py \
  --base_model $BASE_MODEL_PATH \
  --quant_model $QUANT_MODEL_PATH \
  --dataset $DATASET_PATH \
  --nsamples=1024 \
  --val_size=128 \
  --lr=1e-5 \
  --adam_beta1=0.90 \
  --adam_beta2=0.999 \
  --epochs=5 \
  --early_stop=3 \
  --batch_size=1 \
  --microbatch_size=1 \
  --save $DATA_PATH \
  --gradient_checkpointing