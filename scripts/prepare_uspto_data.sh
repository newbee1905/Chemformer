#!/usr/bin/env bash
set -euo pipefail

# Data Preparation Examples for Fine-tuning

echo "Chemformer Data Preparation Examples"
echo "====================================="

# echo "1. Preparing USPTO 5k dataset..."
# python scripts/prepare_fine_tune_data.py \
#   --task uspto \
#   --input data/seq-to-seq_datasets/uspto_50.pickle \
#   --output uspto_5k \
#   --n_samples 5000 \
#   --validate

echo "2. Preparing USPTO 100 dataset..."
python scripts/prepare_fine_tune_data.py \
  --task uspto \
  --input data/seq-to-seq_datasets/uspto_50.pickle \
  --output uspto_100 \
  --n_samples 100 \
  --validate

# echo "3. Preparing full USPTO dataset..."
# python prepare_fine_tune_data.py \
#   --task uspto \
#   --input data/seq-to-seq_datasets/uspto_50.pickle \
#   --output uspto_full \
#   --validate

echo "====================================="
echo "Data preparation completed!"
echo "Available datasets:"
ls -la data/*.pickle
echo "====================================="
