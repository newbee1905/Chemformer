#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

  # data_path=data/seq-to-seq_datasets/uspto_sep.lmdb \
  # model_path=models/fine-tuned/uspto_sep/last_v2.ckpt \
  # model_path="tb_logs/forward_prediction/version_0/checkpoints/last.ckpt" \
  # model_path=\"tb_logs/forward_prediction/version_0/checkpoints/epoch=263-step=2111.ckpt\" \
  # model_path="grpo_logs/forward_prediction/version_0/checkpoints/last.ckpt" \
  # model_path=\"grpo_logs/forward_prediction/version_1/checkpoints/epoch=4-step=339.ckpt\" \
  # model_path=\"tb_logs/forward_prediction/version_21/checkpoints/last.ckpt\" \
  # model_path=models/fine-tuned/uspto_sep/last_v2.ckpt \
	

python -m molbart.inference_score \
  data_path=data/private/private_base \
  vocabulary_path=bart_vocab_downstream.json \
  datamodule=[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB] \
  model_path=\"cycle_finetune_logs/forward_prediction/base/checkpoints/last.ckpt\" \
  task=forward_prediction \
  model_type=graph_bart \
  batch_size=69 \
  n_beams=10 \
	output_score_data="metrics_before_private_base.csv"

# python -m molbart.inference_score \
#   data_path=data/private/private_base \
#   vocabulary_path=bart_vocab_downstream.json \
#   datamodule=[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB] \
#   model_path=\"tb_logs/forward_prediction/version_28/checkpoints/last.ckpt\" \
#   task=forward_prediction \
#   model_type=bart \
#   batch_size=69 \
#   n_beams=10 \
# 	output_score_data="metrics_before_private_base.csv"

# python -m molbart.inference_score \
#   data_path=data/private/private_base \
#   vocabulary_path=bart_vocab_downstream.json \
#   datamodule=[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB] \
#   model_path=\"tb_logs/forward_prediction/version_29/checkpoints/last.ckpt\" \
#   task=forward_prediction \
#   model_type=graph_bart \
#   batch_size=69 \
#   n_beams=10 \
# 	output_score_data="metrics_before_private_base.csv"

# python -m molbart.inference_score \
#   data_path=data/private/private_base \
#   vocabulary_path=bart_vocab_downstream.json \
#   datamodule=[molbart.data.seq2seq_data.UsptoSepDataModuleLMDB] \
#   model_path=\"tb_logs/forward_prediction/version_30/checkpoints/last.ckpt\" \
#   task=forward_prediction \
#   model_type=bart \
#   batch_size=69 \
#   n_beams=10 \
# 	output_score_data="metrics_before_private_base.csv"
