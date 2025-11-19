#!/usr/bin/env bash
set -euo pipefail

# Comprehensive Chemformer Fine-tuning Script
# Supports multiple tasks: backward_prediction, forward_prediction, molecular_optimization

# Configuration
TASK=${1:-backward_prediction}  # backward_prediction, forward_prediction, mol_opt
# TASK=${1:-forward_prediction}  # backward_prediction, forward_prediction, mol_opt
# DATA_PATH=${2:-data/uspto_5k.pickle}
# DATA_PATH=${2:-data/seq-to-seq_datasets/uspto_sep.lmdb}
# DATA_PATH=${2:-data/private/private_base}
DATA_PATH=${2:-data/seq-to-seq_datasets/uspto_50_clean.pickle}
MODEL_PATH=${3:-models/pretrained/v2.ckpt}  # null for from scratch
# MODEL_PATH=${3:-tb_logs/forward_prediction/version_15/checkpoints/last.ckpt}  # null for from scratch
# MODEL_PATH=${3:-models/fine-tuned/uspto_sep/last_v2.ckpt}  # null for from scratch
OUTPUT_DIR=${4:-outputs/fine_tune_$(date +%Y%m%d_%H%M%S)}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Chemformer Fine-tuning"
echo "=========================================="
echo "Task: $TASK"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "=========================================="

# Activate environment if needed
# conda activate chemformer

# Set datamodule based on task
if [[ "$TASK" == "mol_opt" ]]; then
    DATAMODULE="[molbart.data.seq2seq_data.MolecularOptimizationDataModule]"
else
    # DATAMODULE="[molbart.data.seq2seq_data.Uspto50DataModule]"
    # DATAMODULE="[molbart.data.seq2seq_data.UsptoCycleDataModuleLMDB]"
    # DATAMODULE="[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB]"
    DATAMODULE="[molbart.data.seq2seq_data.Uspto50GraphDataModule]"
fi

# Build command
CMD="python -m molbart.fine_tune
	datamodule=\"$DATAMODULE\"
	data_path=\"$DATA_PATH\" 
	model_path=\"$MODEL_PATH\"
	vocabulary_path=bart_vocab_downstream.json
	task=$TASK
	n_epochs=500
	learning_rate=1e-4
	model_type=graph_bart
	schedule=cycle
	optimizer=adamw
	batch_size=32
	acc_batches=1
	augmentation_probability=0.75"

# Add type tokens for USPTO tasks
# if [[ "$TASK" == "backward_prediction" || "$TASK" == "forward_prediction" ]]; then
#     CMD="$CMD +include_type_token=True"
# fi

# Adjust parameters for from-scratch training
if [[ "$MODEL_PATH" == "null" ]]; then
    echo "Training from scratch - adjusting parameters..."
    CMD="$CMD n_epochs=100 learning_rate=3e-4 augmentation_probability=0.8"
fi

echo "Running command:"
echo "$CMD"
echo "=========================================="

# Run training
eval $CMD

echo "=========================================="
echo "Fine-tuning completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Build command
CMD="python -m molbart.fine_tune
  datamodule=\"$DATAMODULE\"
  data_path=\"$DATA_PATH\" 
  model_path=\"$MODEL_PATH\"
  vocabulary_path=bart_vocab_downstream.json
  task=$TASK
  n_epochs=500
  learning_rate=1e-4
	model_type=graph_bart
  schedule=cycle
  optimizer=adamw
  batch_size=77
  acc_batches=1
  augmentation_probability=0.75"

# Add type tokens for USPTO tasks
# if [[ "$TASK" == "backward_prediction" || "$TASK" == "forward_prediction" ]]; then
#     CMD="$CMD +include_type_token=True"
# fi

# Adjust parameters for from-scratch training
if [[ "$MODEL_PATH" == "null" ]]; then
    echo "Training from scratch - adjusting parameters..."
    CMD="$CMD n_epochs=100 learning_rate=3e-4 augmentation_probability=0.8"
fi

echo "Running command:"
echo "$CMD"
echo "=========================================="

# Run training
eval $CMD

echo "=========================================="
echo "Fine-tuning completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
