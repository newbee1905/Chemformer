#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Automated Chemformer Training and Evaluation Script (Multi-Pretrained)
# ==============================================================================
# This script trains multiple model configurations on a series of datasets,
# using multiple base pretrained models. It then evaluates the 'last' and
# 'best' checkpoints for each complete run.

# --- Configuration ---
# Root directory where LMDB dataset folders are located
DATA_ROOT="data/private"
# The task to perform (e.g., forward_prediction)
TASK="forward_prediction"

# --- Define Iteration Options ---
# An associative array of pretrained models to use.
# The KEY is a short identifier for filenames, the VALUE is the path.
declare -A PRETRAINED_MODELS
# PRETRAINED_MODELS["uspto_finetuned"]="models/fine-tuned/uspto_sep/last_v2.ckpt"
PRETRAINED_MODELS["uspto_finetuned"]="tb_logs/forward_prediction/version_15/checkpoints/last.ckpt"
# PRETRAINED_MODELS["v2_base"]="models/pretrained/v2.ckpt"

# An array of all dataset folder names to be processed
# declare -a DATASET_NAMES=("private_base")
# for i in {0..2}; do
# 	DATASET_NAMES+=("private_cv_$i")
# done
# for i in {0..7}; do
# 	DATASET_NAMES+=("private_cv_$i")
# done
declare -a DATASET_NAMES=()
for i in {0..7}; do
	DATASET_NAMES+=("private_cv_$i")
done

# An array of the model types to be trained
declare -a TRAIN_MODEL_TYPES=("cycle_bart")

# --- Main Automation Loop ---
echo "Starting automated training and evaluation process..."

# New outer loop for iterating through different pretrained models
for model_id in "${!PRETRAINED_MODELS[@]}"; do
	PRETRAINED_MODEL_PATH=${PRETRAINED_MODELS[$model_id]}
	
	echo "##################################################################"
	echo "### Starting all runs for Pretrained Model: ${model_id} ###"
	echo "### Path: ${PRETRAINED_MODEL_PATH} ###"
	echo "##################################################################"

	for dataset in "${DATASET_NAMES[@]}"; do
		for train_model in "${TRAIN_MODEL_TYPES[@]}"; do
			echo "================================================================"
			echo ">> STARTING RUN"
			echo ">> Pretrained Model: ${model_id}"
			echo ">> Dataset: ${dataset}"
			echo ">> Model Type for Training: ${train_model}"
			echo "================================================================"

			# --- 1. Training Step ---
			echo
			echo "--- [1/3] Kicking off training... ---"
			DATA_PATH="${DATA_ROOT}/${dataset}"

			# Set datamodule based on the training model type
			if [[ "$train_model" == "cycle_sep_bart" ]]; then
				TRAIN_DATAMODULE="[molbart.data.seq2seq_data.UsptoCycleDataModuleLMDB]"
			else
				TRAIN_DATAMODULE="[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB]"
			fi
			
			echo "Using datamodule: ${TRAIN_DATAMODULE}"
			
			# Execute the training command with the correct pretrained model path
			python -m molbart.fine_tune \
				datamodule="$TRAIN_DATAMODULE" \
				data_path="$DATA_PATH" \
				model_path="$PRETRAINED_MODEL_PATH" \
				vocabulary_path=bart_vocab_downstream.json \
				task=$TASK \
				n_epochs=500 \
				learning_rate=1e-4 \
				model_type=$train_model \
				schedule=cycle \
				optimizer=adamw \
				batch_size=77 \
				acc_batches=1 \
				augmentation_probability=0.75

			# --- 2. Find the Log Directory and Checkpoints ---
			echo
			echo "--- [2/3] Locating checkpoints... ---"
			LOG_DIR=$(ls -td tb_logs/${TASK}/version_* | head -n 1)
			if [[ ! -d "$LOG_DIR/checkpoints" ]]; then
				echo "ERROR: Could not find checkpoints directory in ${LOG_DIR}"
				exit 1
			fi
			echo "Found log directory: ${LOG_DIR}"

			# Define the checkpoints to evaluate
			declare -A checkpoints
			checkpoints["last"]="${LOG_DIR}/checkpoints/last.ckpt"
			
			# Find the 'best' checkpoint (largest step number)
			best_ckpt=$(find "${LOG_DIR}/checkpoints/" -name "epoch=*-validation_loss=*.ckpt" | sort -t '=' -k 3 -n | head -n 1)
			if [[ -f "$best_ckpt" ]]; then
				checkpoints["best"]="$best_ckpt"
				echo "Found 'best' checkpoint: $(basename "${checkpoints['best']}")"
			else
				echo "WARNING: Could not find a 'best' checkpoint (e.g., epoch=*-step=...). Skipping."
			fi

			# --- 3. Evaluation Step ---
			echo
			echo "--- [3/3] Starting evaluation... ---"

			for ckpt_type in "${!checkpoints[@]}"; do
				ckpt_path=${checkpoints[$ckpt_type]}
				echo
				echo "--> Evaluating '${ckpt_type}' checkpoint: $(basename "$ckpt_path")"
				
				# Dynamically name the output metrics file, now including the pretrained model ID
				OUTPUT_METRICS_FILE="metrics_${model_id}_${train_model}_${dataset}_${ckpt_type}.csv"
				OUTPUT_SMILES_FILE="smiles_${model_id}_${train_model}_${dataset}_${ckpt_type}.json"
				
				# As requested, evaluation ALWAYS uses 'bart' model_type and UsptoSepDataModuleLMDB
				EVAL_DATAMODULE="[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB]"
				EVAL_MODEL_TYPE="bart"

				# Execute the evaluation command
				python -m molbart.inference_score \
					data_path="$DATA_PATH" \
					model_path=\"$ckpt_path\" \
					vocabulary_path=bart_vocab_downstream.json \
					datamodule=$EVAL_DATAMODULE \
					task=$TASK \
					model_type=$EVAL_MODEL_TYPE \
					batch_size=69 \
					n_beams=10 \
					output_score_data="$OUTPUT_METRICS_FILE" \
					output_sampled_smiles="$OUTPUT_SMILES_FILE"

				echo "--> Evaluation complete. Metrics saved to ${OUTPUT_METRICS_FILE}"
			done # End of checkpoint loop

			echo
			echo ">> Run for '${train_model}' on '${dataset}' (from '${model_id}') finished."
			echo "================================================================"
			echo

		done # End of model type loop
	done # End of dataset loop
done # End of pretrained model loop

echo "All automated tasks have been completed!"
