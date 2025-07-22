#!/bin/bash

# python -m molbart.pretrain \
#   datamodule=[molbart.data.mol_data.ZincDataModule] \
#   data_path=data/zinc \
#   model_type=bart \
#   learning_rate=1.0 \
#   schedule=transformer \
#   n_epochs=100 \
#   batch_size=64 \
#   d_model=1024 \
#   n_layers=8 \
#   n_heads=16 \
#   d_feedforward=4096 \
#   n_gpus=4 \
#   task=mask_aug

export PYTHONPATH=$PYTHONPATH:.

python -m molbart.pretrain \
  datamodule=[molbart.data.datamodules.ChemblDataModule] \
  data_path=data/chembl_27_xsmall.pickle \
  model_path=models/bart/pre_train.ckpt \
  vocabulary_path=bart_vocab_downstream.json \
  n_epochs=100 \
  batch_size=64 \
  acc_batches=4

