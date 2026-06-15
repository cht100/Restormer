#!/usr/bin/env bash

CONFIG=$1

#python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=4321
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO

torchrun --nproc_per_node=2 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         -m basicsr.train \
         -opt ./Deraining/Options/Deraining_Restormer.yml \
         --launcher pytorch