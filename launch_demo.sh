#!/bin/bash
# simpletuner_run.sh

# Exit on error
set -e

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpletuner

# Set the CUDA device
export CUDA_VISIBLE_DEVICES=2

# Run the script
python lora_inf.py
