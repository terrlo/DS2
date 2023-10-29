#!/bin/bash

set -e
set -x

data_dir="./dataset/mvtec_train/"
output_dir="./output/mvtec"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_port 12300 --nproc_per_node=2 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    --model CutPaste \
    --seed 2 \
    --cutpaste_category all \
    --note "note" \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##