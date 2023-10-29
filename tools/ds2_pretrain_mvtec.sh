#!/bin/bash

set -e
set -x

data_dir="./dataset/mvtec_train/"
output_dir="./output/mvtec"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_port 12301 --nproc_per_node=2 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    --dataset MVTecAD \
    --dataset_portion 1.0 \
    --image-size 224 \
    \
    --lr-scheduler step \
    --lr-decay-epochs 120 160 200 \
    --optimizer lars \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --crop 0.08 \
    \
    --instance_loss_weight 0.0 \
    --instance_loss_func DistAug_SimCLR \
    --dense_loss_weight 1.0 \
    --dense_loss_func DS2 \
    \
    --model DS2 \
    --epochs 400 \
    --save-freq 100 \
    --batch-size 128 \
    --base-lr 2.0 \
    --seed 2 \
    --note """note""" \