# Dual Scale Dual Similarity (DS2)

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

## MVTec LOCO dataset preparation:

- visit https://www.mvtec.com/company/research/datasets/mvtec-loco, and fill out the form required in the website to download the dataset
- once downloaded, unzip the file, rename the outermost folder name to `mvtecloco`, and move the files to folder `DS2/dataset/`, such that the **breakfast_box** category is located at `DS2/dataset/mvtecloco/breakfast_box`

## KSDD2 dataset preparation:

- download the dataaset: `wget https://go.vicos.si/kolektorsdd2 -O KSDD2.zip`
- unzip the file: `unzip KSDD2.zip`
- move the `train` and `test` folders to `DS2/dataset/KSDD2/`

## MTD dataset preparation:

- download the dataset: `git clone https://github.com/abin24/Magnetic-tile-defect-datasets..git`
- rename the folder name from `Magnetic-tile-defect-datasets.` to `MTD`
- move the `MTD` folder to `DS2/dataset/`

## Run pretraining code for DS2 on MVTec (Stage 1):

- To run the pretraining code for DS2, execute `./tools/ds2_pretrain_mvtec.sh`. The default setting requires two GPUs (preferably A100-40GB and above). The seed range is [1,5]
- The output log (including model checkpoints) will be stored in folder **_output/mvtec\_$TIMESTAMP/_**

## Run evaluation code for DS2 on MVTec (Stage 2):

- After pretraining, execute `./tools/ds2_eval_mvtec.sh` to perform anomaly detection on test split. The default setting requires one GPU.
  - Inside file `./tools/ds2_eval_mvtec.sh`, set `pretrained_model_dir` to the checkpoint models' folder **_output/mvtec\_$TIMESTAMP/_**
- The evaluation output will be stored in folder **_logs/_**

## Run pretraining code for CutPaste\_(3-way, one-for-all) on MVTec:

- To run the pretraining code for CutPaste, execute `./tools/cutpaste_pretrain_mvtec.sh`. The default setting requires two GPUs (preferably A100-40GB and above). The seed range is [1,5]
- The output log (including model checkpoints) will be stored in folder **output/mvtec\_$TIMESTAMP_cutpaste/**

## Run evaluation code for CutPaste\_(3-way, one-for-all) on MVTec:

- After pretraining, execute `./tools/cutpaste_eval_mvtec.sh` to perform anomaly detection on test split. The default setting requires one GPU.
  - Inside file `./tools/cutpaste_eval_mvtec.sh`, set `pretrained_model_dir` to the checkpoint models' folder **output/mvtec\_$TIMESTAMP_cutpaste/**
- The evaluation output will be stored in folder **_logs/_**

## Run evaluation code for DS2 on MVTec LOCO:

- After pretraining, execute `./tools/ds2_eval_loco.sh`. The default setting requires one GPU.
  - Inside file `./tools/ds2_eval_loco.sh`, set `pretrained_model_dir` to the checkpoint models' folder **_output/mvtec\_$TIMESTAMP/_**
- The evaluation output will be stored in folder **_logs/_**

## Run evaluation code for DS2 on KSDD2:

- After pretraining, execute `./tools/ds2_eval_ksdd2.sh`. The default setting requires one GPU.
  - Inside file `./tools/ds2_eval_ksdd2.sh`, set `pretrained_model_dir` to the checkpoint models' folder **_output/mvtec\_$TIMESTAMP/_**
- The evaluation output will be stored in folder **_logs/_**

## Run evaluation code for DS2 on MTD:

- After pretraining, execute `./tools/ds2_eval_mtd.sh`. The default setting requires one GPU.
  - Inside file `./tools/ds2_eval_mtd.sh`, set `pretrained_model_dir` to the checkpoint models' folder **_output/mvtec\_$TIMESTAMP/_**
- The evaluation output will be stored in folder **_logs/_**

## Acknowledgement

The main architecture is adapted from https://github.com/zdaxie/PixPro (Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning)

The implementation of DistAug and RotPred is adapted from https://github.com/google-research/deep_representation_one_class (LEARNING AND EVALUATING REPRESENTATIONS FOR DEEP ONE-CLASS CLASSIFICATION)

The implementation of CutPaste is adapted from https://github.com/Runinho/pytorch-cutpaste
