#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python -m molbart.inference_score \
  data_path=data/uspto_sep.pickle \
  model_path=models/fine-tuned/uspto_sep/last_v2.ckpt \
  vocabulary_path=bart_vocab_downstream.json \
  datamodule=[molbart.data.seq2seq_data.UsptoSepDataModule] \
  task=forward_prediction \
  model_type=bart \
  batch_size=64 \
  n_beams=10

