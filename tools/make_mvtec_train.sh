#!/bin/bash

set -e
set -x

for type in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
do
  mkdir $PROJ_ABS_PATH/dataset/mvtec_train/$type/
  cp $PROJ_ABS_PATH/dataset/mvtec/$type/train/good/* $PROJ_ABS_PATH/dataset/mvtec_train/$type/
done