# Dual Scale Dual Similarity (DS2)

## Source code repository:

https://github.com/terrlo/DS2

## Acknowledgement

The main architecture of our codes builds upon the publicly available codes from https://github.com/zdaxie/PixPro (Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning)

The part of codes regarding DistAug and RotPred are adapted from https://github.com/google-research/deep_representation_one_class (LEARNING AND EVALUATING REPRESENTATIONS FOR DEEP ONE-CLASS CLASSIFICATION)

## Supported platform:

Linux

## Package installation:

`conda install` the following packages in the recommended order:

- `pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch`
- `scikit-learn`
- `python=3.8.8`
- `opencv -c conda-forge`
- `termcolor -c omnia`
- `pillow=9.1.0`

## MVTec dataset preparation:

Follow the steps to create **mvtec** dataset (for evaluation stage) and **mvtec_train** dataset (for pretraining stage)

- create the **dataset** folder inside the project root: `mkdir dataset`
- move into the folder: `cd dataset`
- create the **mvtec** folder: `mdkir mvtec`
- move into the folder: `cd mvtec`
- Download MVTec AD dataset: `wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz`
- unzip: `tar -xf mvtec_anomaly_detection.tar.xz`
- remove zip file: `rm mvtec_anomaly_detection.tar.xz`
- move to parent folder (**dataset/**): `cd ..`
- create the **mvtec_train** folder: `mkdir mvtec_train`
- move to parent folder (project root): `cd ..`
- run the command `./tools/make_mvtec_train.sh` to create the **mvtec_train** dataset for pretraining (Note: replace the `$PROJ_ABS_PATH` to the absolute path of your project on your local machine)

## Run pretraining code (Stage 1):

- To run the pretraining code, execute `./tools/mvtec_pretrain.sh`. The default setting requires two GPUs.
- The output log (including model checkpoints) will be stored in folder **_output/mvtec\_$TIMESTAMP/_**

## Run evaluation code (Stage 2):

- After pretraining, execute `./tools/mvtec_eval.sh` to perform anomaly detection on test split. The default setting requires one GPU.
  - Inside file **_mvtec_eval.sh_**, set `pretrained_model_dir` to the checkpoint models' folder **_output/mvtec\_$TIMESTAMP/_**
- The evaluation output will be stored in folder **_logs/_**
