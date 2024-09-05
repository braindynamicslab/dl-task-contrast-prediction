# dl-task-contrast-prediction
 Code for the work entitled "Predicting Task Activation Maps from Resting-State Functional Connectivity using Deep Learning" by Madsen et al.

## Referenced Work
Soren J. Madsen, Lucina Q. Uddin, Jeannette A. Mumford, Deanna M. Barch, Damien A. Fair, Ian H. Gotlib, Russell A. Poldrack, Amy Kuceyeski, Manish Saggar. **Predicting Task Activation Maps from Resting-State Functional Connectivity using Deep Learning**

## Overview

This project contains the source code for several models to be used in predicting task activation maps from resting-state fMRI data. BrainSurfCNN is a surface-based convolutional neural network implemented in Ngo et al., 2022 (https://github.com/ngohgia/brain-surf-cnn). BrainSERF makes incremental changes to this model to include Squeeze-Excitation attention and modified activation functions. BrainSurfGCN uses a graph convolution approach instead of surface-based convolution.

Note: Much of the work in this code has been adapted from [BrainSurfCNN](https://github.com/ngohgia/brain-surf-cnn) which is adapted from the [UGSCNN](https://github.com/maxjiang93/ugscnn).

1. [data](./data) folder contains the surface mesh templates, medial-wall mask and subject IDs from the Human Connectome Project (HCP) S1200 used in our experiments.
2. [model](./model) folder contains BrainSurfCNN, BrainSERF, and BrainSurfGCN source code.
3. [preprocess](./preprocess) folder contains functions needed for preprocessing the surface data.
4. [utils](./utils) folder contains utility functions to run the experiment and perform some portions post-hoc evaluation.
5. [train](./train) folder contains training functions to train each stage of the model (MSE or fine-tuning) and save the model parameters.
6. [test](./test) folder contains functions to test the trained models and save the predictions.
7. [posthoc_analysis](.posthoc_analysis) folder contains Jupyter notebooks for computing model metrics and creating figures used in the paper. Please note: in order to recreate figures, it may require some playing around with the plotting parameters especially for the figures that are plotted on brains.
----

## Getting Started

1. Set up your environment with the packages in `requirements.txt` in Python 3.9:

2. Download HCP Workbench [https://www.humanconnectome.org/software/get-connectome-workbench](https://www.humanconnectome.org/software/get-connectome-workbench) for data preprocessing.

3. Download HCP S1200 and HCP Retest dataset [https://db.humanconnectome.org/](https://db.humanconnectome.org), which are used in our experiments.

4. HCP1200 Parcellation+Timeseries+Netmats (PTN) [https://db.humanconnectome.org/data/projects/HCP_1200](https://db.humanconnectome.org/data/projects/HCP_1200) data are also needed for computing the resting-state fingerprints.

5. Run data preprocessing with the scripts under `preprocess` folder.

6. Run training and prediction with the scripts under the `train` and `predict` folders. Note: each script is designed to work with one type of model rather than that being an input argument.

Example Usage:
```
PROJECT_DIR=/home/users/sjmadsen/dl-task-contrast-prediction

NUM_ICS=100
NUM_SAMPLES=8
NUM_VAL_SUBJ=5

RSFC_DIR=/path/to/rsfc_data
CONTRASTS_DIR=/path/to/joint_contrasts
SUBJ_LIST_FILE=/path/to/HCP_train_val_subj_ids.csv
MESH_TEMPLATES_DIR=/path/to/fs_LR_mesh_templates
MESH_PATH=$MESH_TEMPLATES_DIR/icosphere_2.pkl
OUTPUTS_DIR=/where/to/store/the/model.pth

python3 -u train_gnn.py \
       --gpus 0 \
       --ver gnn_mse_larger \
       --n_samples_per_subj $NUM_SAMPLES\
       --loss mse \
       --subj_list $SUBJ_LIST_FILE \
       --rsfc_dir $RSFC_DIR \
       --contrast_dir $CONTRASTS_DIR \
       --mesh_dir $MESH_PATH \
       --save_dir $OUTPUTS_DIR \
       --n_val_subj $NUM_VAL_SUBJ \
       --n_channels_per_hemi $NUM_ICS
```
Note: BrainSurfGCN's `--mesh_dir` argument requires a path to the `icosphere_2.pkl` file whereas BrainSurfCNN and BrainSERF require the entire directory of `fs_LR_mesh_templates/` for the model parameters.

## Having Issues?

Email Soren Madsen at soren dot j dot madsen at gmail dot com.
