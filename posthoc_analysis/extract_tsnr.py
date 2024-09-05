import os
import nibabel.gifti as gi
import numpy as np

class args:
    input_dir = '/oak/stanford/groups/saggar/rdoc-compute/hcp/1200/brainsurf_cnn/1_rsfmri'
    subj_ids = '../data/MICCAI2020/HCP_test_retest_subj_ids.csv'
    output_dir = '/oak/stanford/groups/saggar/rdoc-compute/hcp/1200/brainsurf_cnn/test_tsnr'

def extract_ts_from_mesh(mesh_file, num_ts=1200):
    mesh = gi.read(mesh_file)
    data = []

    if (num_ts != len(mesh.darrays)):
        print(len(mesh.darrays))
    assert(num_ts == len(mesh.darrays))
    for i in range(num_ts):
        data.append(mesh.darrays[i].data)
    data = np.asarray(data).T
    return data

subj_ids = np.genfromtxt(args.subj_ids, dtype='<U20')

for j in range(len(subj_ids)):
    subj_id = subj_ids[j]
    subj_id = f'sub-{subj_id}'
    print(j+1, "/", len(subj_ids), ":", subj_id)

    output_file = os.path.join(args.output_dir, f'{subj_id}_tsnr.npy')

    lh_subj_rest1_lr_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.L.func.gii')
    lh_subj_rest1_rl_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.L.func.gii')
    lh_subj_rest2_lr_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.L.func.gii')
    lh_subj_rest2_rl_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.L.func.gii')

    rh_subj_rest1_lr_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.R.func.gii')
    rh_subj_rest1_rl_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.R.func.gii')
    rh_subj_rest2_lr_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.R.func.gii')
    rh_subj_rest2_rl_file = os.path.join(args.input_dir, subj_id, f'{subj_id}_rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.R.func.gii')
    #subj_node_ts_file = os.path.join(args.node_ts_dir, "%s.txt" % subj_id)

    if os.path.exists(lh_subj_rest1_lr_file) and os.path.exists(lh_subj_rest1_rl_file) and os.path.exists(lh_subj_rest2_lr_file) and os.path.exists(lh_subj_rest2_rl_file):
        try:
            lh_subj_rest1_lr_data = extract_ts_from_mesh(lh_subj_rest1_lr_file)
            lh_subj_rest1_rl_data = extract_ts_from_mesh(lh_subj_rest1_rl_file)
            lh_subj_rest2_lr_data = extract_ts_from_mesh(lh_subj_rest2_lr_file)
            lh_subj_rest2_rl_data = extract_ts_from_mesh(lh_subj_rest2_rl_file)

            rh_subj_rest1_lr_data = extract_ts_from_mesh(rh_subj_rest1_lr_file)
            rh_subj_rest1_rl_data = extract_ts_from_mesh(rh_subj_rest1_rl_file)
            rh_subj_rest2_lr_data = extract_ts_from_mesh(rh_subj_rest2_lr_file)
            rh_subj_rest2_rl_data = extract_ts_from_mesh(rh_subj_rest2_rl_file)

            lh_data = np.concatenate((lh_subj_rest1_lr_data, lh_subj_rest1_rl_data, lh_subj_rest2_lr_data, lh_subj_rest2_rl_data), axis=1)
            rh_data = np.concatenate((rh_subj_rest1_lr_data, rh_subj_rest1_rl_data, rh_subj_rest2_lr_data, rh_subj_rest2_rl_data), axis=1)
            lh_data = lh_data.mean(1) / (lh_data.std(1) + 1e-9)
            rh_data = rh_data.mean(1) / (rh_data.std(1) + 1e-9)
            output_data = (lh_data, rh_data)
            print(lh_data.shape, np.max(lh_data))
            np.save(output_file, output_data)
        except:
            print("Error", subj_id)
    break