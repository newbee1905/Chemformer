#!/bin/bash

# python -m molbart.fine_tune \  
#   "datamodule=[molbart.data.seq2seq_data.Uspto50DataModule]" \
#   data_path=data/uspto_50.pickle \
#   model_path=models/bart/span_aug.ckpt \
#   vocabulary_path=bart_vocab_downstream.json \
#   task=backward_prediction \
#   n_epochs=100 \
#   learning_rate=0.001 \
#   schedule=cycle \
#   batch_size=64 \
#   acc_batches=4 \
#   augmentation_probability=0.5

TASK=${1:-forward_prediction}
DATA_PATH=${3:-data/private/private_base}
MODEL_PATH=${3:-finetuned_logs/forward_prediction/base/checkpoints/last.ckpt}
OUTPUT_DIR=${4:-outputs/fine_tune_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUTPUT_DIR"

DATAMODULE="[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB]"

CMD="python -m molbart.rl
  datamodule=\"$DATAMODULE\"
  data_path=\"$DATA_PATH\" 
  model_path=\"$MODEL_PATH\"
  vocabulary_path=bart_vocab_downstream.json
  task=$TASK
  n_epochs=100
  learning_rate=1e-4
	model_type=bart
  batch_size=8
  acc_batches=1
  augmentation_probability=0.75
	weight_decay=0.1
	n_beams=5
	rl.g_samples=3"

eval $CMD
