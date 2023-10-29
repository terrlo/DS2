#!/bin/bash

set -e
set -x

# change "pretrained_model_dir" to your pretrained model's folder name
pretrained_model_dir="./output/mvtec_TIMESTAMP"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 other_eval.py \
    --pretrained_model_dir ${pretrained_model_dir} \
    --gde_smp_size 1000 \
    --gde_count one \
    --ckpt_epoch 400 \
    --dataset KSDD2 \
    --note "note" \
    # --qualitative \