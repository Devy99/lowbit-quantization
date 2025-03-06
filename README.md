# [Replication Package] Quantizing Large Language Models for Code Generation: A Differentiated Replication

## Introduction
This repository contains the scripts needed to replicate the findings of our study: _"Quantizing Large Language Models for Code Generation: A Differentiated Replication"_. In the following sections, we discuss the code, datasets, and other files included in the replication package.

## Contents 
1. [Requirements](#requirements)
2. [Datasets and other materials](#datasets-and-other-materials)
3. [Replication of the results](#replication-of-the-results)

## Requirements
### Python Dependencies
The [requirements.txt](requirements.txt) file contains all the dependencies required to run our scripts. We recommend installing these on your machine using a separate virtual environment, as follows:
  ```sh
   # Create a new virtual environment and activate it
   python3 -m venv venv
   source venv/bin/activate

   # Install the packages from the requirements.txt file
   pip3 install -r requirements.txt
   ```

In order to run the quantized models, you must also install the AQLM Python dependency.
You can specify which dependency to install based on the target inference hardware, which can be CPU or GPU, or both as follows:
  ```sh
   # Install AQLM dependencies
   pip3 install aqlm[gpu,cpu]
   ```

More information can be found in the official [AQLM repository](https://github.com/Vahe1994/AQLM).

### Software Dependencies
Our scripts rely on two code repositories:

- [AQLM](https://github.com/Vahe1994/AQLM), used for quantizing and fine-tuning our models. 
- [MultiPL-E](https://github.com/nuprl/MultiPL-E), for evaluating the base models and their quantized versions on Java and Python benchmarks. 

Both repositories are continuously evolving. To ensure the reproducibility of our results, we rely on commit version `f13ca20` for AQLM and `19a2567` for MultiPL-E. Furthermore, we added new functionality to these repositories. For example, we updated  AQLM to use the "Mixed" and "Code" calibration datasets, in addition to the "Random" one.

### Hardware 
The hardware required to run these scripts is entirely dependent on the models you want to analyze. We evaluated the quantized models on a consumer card (NVIDIA GeForce RTX 3090), while we used an A100 GPU (80GB) for model quantization and fine-tuning, as well as to evaluate larger models at their original precision.

## Datasets and other materials
All the results and materials from our work are available in our [Zenodo repository](https://doi.org/10.5281/zenodo.13752774).

In particular, you can find the following folders: 
- **datasets**: includes the random, mixed, and code calibration datasets analyzed in RQ2 and the Python/Java McEval benchmarks.
- **predictions**: we provide completions of all analyzed models for each experiment and benchmark in MultiPL-E-compatible files. Each folder contains `*.json.gz` files with all model's predictions for each task, as well as `*.results.json.gz` files with the test results for each completion. 
- **results**: contains the accuracies of the evaluated models as well as the results of statistical tests on both MultiPL-E and McEval benchmarks.


## Replication of the results
Below, we provide a step-by-step guide for running our code.

### Initialization 
As a first step, run the [init.sh](init.sh) script. 
```sh
# Setup AQLM and MultiPL-E tools
bash init.sh
```
This script downloads AQLM and MultiPL-E repositories, updates their version to the ones discussed in [Software Dependencies](#software-dependencies) section, and adds additional functionality needed for our analyses. 

### Download calibration datasets 
Under the `1-calibration-datasets` folder, we include the [1_download_datasets.sh](./1-calibration-datasets/1_download_datasets.sh) script for generating the calibration datasets.

To run this script, you must provide the name of the target model (on HuggingFace) or the path where checkpoints are stored, as shown below:
```sh
# Download calibration dataset
bash 1_download_datasets.sh <MODEL_NAME_OR_PATH>
```

In output, it will provide three different datasets: `red_pajama_calibration_sample` (the random dataset), `mixed_sample` (the mixed dataset), and `github_code_sample` (the code dataset).

### Quantization
To quantize models with AQLM, you find the [quantize.sh](./2-quantize-models/quantize.sh) script in the `2-quantize-models` folder.

The script requires the following arguments as input:
```sh
# Quantize the model
bash quantize.sh <MODEL_NAME_OR_PATH> <CALIBRATION_TYPE> <OUTPUT_FOLDER>  <NUM_CODEBOOKS> <NBITS_PER_CODEBOOK> <IN_GROUP_SIZE>
```
- **Model name or path:** the name of the model to quantize. It can be either a model on HuggingFace or the path where the model is stored in the local machine
- **Calibration type:** the name of the calibration dataset to use. You have three options for the type of dataset: `pajama` (for the random dataset), `code_technical_language` (for the mixed dataset), and `github_code` (for the code dataset).
- **Output folder:** path where to store the model's quantized layers.
- **Num codebooks:** number of codebooks (indicated as M in the paper)
- **Nbits per codebook:** number of bits for each codebook (indicated as B in the paper)
- **In group size:** number of weights to process together (indicated as g in the paper)


For example, if we want to quantize CodeLlama 7B to 2-bit precision using a random calibration dataset, we must provide the following arguments:
```sh
# Quantize the model
bash quantize.sh   codellama/CodeLlama-7b-hf   pajama   output   1   15   8
```

Finally, it is necessary to convert the files saved in `OUTPUT_FOLDER` into a format compatible with the HuggingFace transformers library, using the AQLM script `convert_to_hf.py` as follows:

```sh
# Convert quantized layer to a .bin format
python AQLM/convert_to_hf.py <BASE_MODEL_NAME_OR_PATH> <QUANTIZATION_FOLDER> <MODEL_OUTPUT_FOLDER> --save_tokenizer
```
- **Base model name or path:** name or path of the full-precision model
- **Quantization folder:** `OUTPUT_FOLDER` path, where quantized layers are stored 
- **Model output folder:** path of the folder where to store the final quantized model


### End-to-end fine-tuning
After quantizing the model, you can run end-to-end fine-tuning with the script provided under `3-finetuning` folder.

Below, are the parameters to provide in input to the script:

```sh
# Fine-tune the quantized model
bash finetune.sh <QUANTIZATION_FOLDER> <BASE_MODEL_NAME_OR_PATH> <CALIBRATION_TYPE> <OUTPUT_FOLDER>
```
- **Quantization folder:** the path where the layers of the quantized model are stored.
- **Base model name or path:** name or path of the full-precision model (teacher)
- **Calibration type:** the name of the calibration dataset to use. As before, you have three options for the type of dataset: `pajama` (for the random dataset), `code_technical_language` (for the mixed dataset), and `github_code` (for the code dataset).
- **Output folder:** path where to store the fine-tuned model.

### Evaluation
To analyze the models with MultiPL-E, run one of the scripts in the `4-run-benchmarks` folder, specifically [1_run_benchmark_fp16.sh](./4-run-benchmarks/1_run_benchmark_fp16.sh) or [2_run_benchmark_aqlm.sh](./4-run-benchmarks/2_run_benchmark_aqlm.sh). The first must be used to evaluate base models with their original precision, while the second is tailored for inference on AQLM quantized models.

The parameters to provide to the scripts are as follows:

```sh
# Evaluate a model in its original precision
bash 1_run_benchmark_fp16.sh <LANGUAGE_INDEX> <MODEL_NAME_OR_PATH>

# Evaluate a model quantized with AQLM
bash 2_run_benchmark_aqlm.sh <LANGUAGE_INDEX> <MODEL_NAME_OR_PATH>
```
- **Language index:** index of the language to analyze. Select `0` for Java and `1` for Python 
- **Model name or path:** model name on HuggingFace or path where the checkpoints are stored in the local machine


For example, to run the MultiPL-E benchmark for Java code on CodeLlama 7B, you must provide the following arguments:
```sh
# Evaluate CodeLlama 7B with its original precision
bash multipl-e/1_run_benchmark_fp16.sh  0  codellama/CodeLlama-7b-hf
```

This script will output the predictions of the model under the `results_cleaned` folder. 