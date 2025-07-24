# Training Models

## Referenced Work
Soren J. Madsen, Young-Eun Lee, Lucina Q. Uddin, Jeannette A. Mumford, Deanna M. Barch, Damien A. Fair, Ian H. Gotlib, Russell A. Poldrack, Amy Kuceyeski, Manish Saggar. **Predicting Task Activation Maps from Resting-State Functional Connectivity using Deep Learning** [[bioarXiv]](https://www.biorxiv.org/content/10.1101/2024.09.10.612309v2)

----
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

----

## Training Parameters
Check [utils/parser.py](https://github.com/braindynamicslab/dl-task-contrast-prediction/blob/main/utils/parser.py)

