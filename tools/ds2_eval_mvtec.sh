#!/bin/bash

set -e
set -x

# change "pretrained_model_dir" to your pretrained model's folder name
pretrained_model_dir="./output/mvtec_TIMESTAMP"
mvtec_dataset_dir="./dataset/mvtec/"
log_dir="./logs/"
qualitative_dir="./qualitative/"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 mvtec_eval.py \
    --pretrained_model_dir ${pretrained_model_dir} \
    --mvtec_dataset_dir ${mvtec_dataset_dir} \
    --log_dir ${log_dir} \
    --qualitative_dir ${qualitative_dir} \
    \
    --patch_size 32 \
    --patch_stride 4 \
    --gde_smp_size 350 \
    --resized_image_size 256 \
    \
    --category all \
    --ckpt_epoch 300 \
    --note "note" \
    # --qualitative \