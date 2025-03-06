#!/bin/bash

# ========== Setup the benchmark tools ========== #

# Setup MultiPL-E docker image
TAG="multipl-e-eval"
IMAGE="ghcr.io/nuprl/multipl-e-evaluation:latest"
if [ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]; then
    echo "Pulling and tagging Multipl-E image..."
    docker pull $IMAGE
    docker tag $IMAGE $TAG
else
    echo "Multipl-E image already set up. Continuing..."
fi


# Verify that there are enough GPUs available to run the benchmark on separate GPUs
MIN_GPUS=2
GPU_NUMBER=$1
if [ -z "$GPU_NUMBER" ] || [ "$GPU_NUMBER" -lt 0 ] || [ "$GPU_NUMBER" -gt $(($MIN_GPUS - 1)) ]; then
    echo "Please provide a valid GPU number (0-1) as parameter."
    exit 1
fi

# ========== Setup the benchmark parameters ========== #
LANGUAGES=("java" "py")
LANGUAGE=${LANGUAGES[$GPU_NUMBER]}

MODEL_NAME=$2
MODEL_LABEL=$(echo $MODEL_NAME | cut -d'/' -f 2)

BENCHMARK_DATASET="mceval_${LANGUAGE}.jsonl"
BATCH_SIZE=1
MAX_TOKENS=1500
TEMPERATURE=0.2
COMPLETION_LIMIT=20

LABEL="aqlm"
UTILS_DIR="../utils/"

# Create output directory
RESULTS_DIR="./results/$MODEL_LABEL"
mkdir -p $RESULTS_DIR
OUTPUT_DIR="${RESULTS_DIR}/${LANGUAGE}_benchmark_temperature_${TEMPERATURE}_$LABEL"
mkdir -p $OUTPUT_DIR


# ========== Running model evaluation ========== #

echo "Running benchmark with the following parameters:"
echo "Model name: $MODEL_NAME, Benchmark dataset: $BENCHMARK_DATASET, Language: $LANGUAGE, Temperature: $TEMPERATURE, Batch size: $BATCH_SIZE, Completion limit: $COMPLETION_LIMIT, Max tokens: $MAX_TOKENS, Label: $LABEL"


# Run the model generation script
echo "Running model generation script..."

python3 -u $UTILS_DIR/MultiPL-E/automodel.py \
        --name $MODEL_NAME \
        --use-local \
        --dataset $BENCHMARK_DATASET \
        --lang $LANGUAGE \
        --temperature $TEMPERATURE \
        --batch-size $BATCH_SIZE \
        --completion-limit $COMPLETION_LIMIT \
        --output-dir-prefix $OUTPUT_DIR \
        --max-tokens $MAX_TOKENS 

# Sanitize the generated completions
NEW_RESULTS_DIR=$(echo $RESULTS_DIR | awk -F'/' '{OFS="/"; $2=$2"_cleaned"; print $0}')
CLEANED_OUTPUT_DIR="${NEW_RESULTS_DIR}/${LANGUAGE}_benchmark_temperature_${TEMPERATURE}_$LABEL"
python3 -u $UTILS_DIR/clean_completions.py --input_dir $OUTPUT_DIR --language $LANGUAGE --output_dir $CLEANED_OUTPUT_DIR

# Run the evaluation script
echo "Running evaluation script..."

docker run --rm --network none -v ./$CLEANED_OUTPUT_DIR:/$CLEANED_OUTPUT_DIR:rw multipl-e-eval --dir /$CLEANED_OUTPUT_DIR --output-dir /$CLEANED_OUTPUT_DIR --recursive

# Print the results
echo "Printing the results..."

TEMPERATURE_LABEL=$(echo $TEMPERATURE | sed 's/\./_/g')
python3 -u $UTILS_DIR/MultiPL-E/pass_k.py ./$CLEANED_OUTPUT_DIR/* | tee $CLEANED_OUTPUT_DIR/pass_result_temp_${TEMPERATURE_LABEL}.csv
