import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.transforms.face_to_edge import FaceToEdge

class MultiContrastGraphDataset(Dataset):
    # Randomly samples an input and a group task contrast
    def __init__(self, subj_ids, rsfc_dir, contrast_dir, mesh_path, num_samples=8):
        self.rsfc_dir = rsfc_dir
        self.contrast_dir = contrast_dir
        self.subj_ids = subj_ids
        self.num_samples = num_samples
        self.mesh_path = mesh_path
        self.f2e = FaceToEdge()
        # self.task_samples = task_samples


    def __getitem__(self, index):
        subj = self.subj_ids[index]

        sample_id = np.random.randint(0, self.num_samples)
        rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_sub-%s_sample%d_rsfc.npy" % (subj, sample_id))
	#rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_%s_sample%d_rsfc.npy" % (subj, sample_id))
        subj_rsfc_data = np.load(rsfc_file).T # shape = V x 2*ICs

        subj_task_data = np.load(os.path.join(self.contrast_dir, "%s_joint_LR_task_contrasts.npy" % subj))

        data = Data()
        mesh = np.load(self.mesh_path, allow_pickle=True)
        data.face = torch.tensor(mesh['F'].T)
        data.num_nodes = mesh['V'].shape[0]
        data = self.f2e(data)

        # Adds positional awareness
        input_data = np.concatenate([subj_rsfc_data, mesh['V']], axis=-1) # shape = V x 2*ICs + 3

        data.x = torch.FloatTensor(input_data)
        target = torch.cuda.FloatTensor(subj_task_data)

        return data.cuda(), target
    
    def __len__(self):
        return len(self.subj_ids)

class MultipleSampleMeshDataset(Dataset):
    def __init__(self, subj_ids, rsfc_dir, contrast_dir, num_samples=8):
        self.rsfc_dir = rsfc_dir
        self.contrast_dir = contrast_dir
        self.subj_ids = subj_ids
        self.num_samples = num_samples

    def __getitem__(self, index):
        subj = self.subj_ids[index]

        sample_id = np.random.randint(0, self.num_samples)
        rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_sub-%s_sample%d_rsfc.npy" % (subj, sample_id))
	#rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_%s_sample%d_rsfc.npy" % (subj, sample_id))
        subj_rsfc_data = np.load(rsfc_file)

        subj_task_data = np.load(os.path.join(self.contrast_dir, "%s_joint_LR_task_contrasts.npy" % subj))

        return torch.cuda.FloatTensor(subj_rsfc_data) , torch.cuda.FloatTensor(subj_task_data)

    def __len__(self):
        return len(self.subj_ids)
    
class GroupContrastMultipleSampleMeshDataset(Dataset):
    def __init__(self, subj_ids, rsfc_dir, contrast_dir, group_contrast_dir, task_id, num_samples=8):
        self.rsfc_dir = rsfc_dir
        self.contrast_dir = contrast_dir
        self.subj_ids = subj_ids
        self.num_samples = num_samples
        self.group_contrast_dir = group_contrast_dir
        self.task_id = task_id


    def __getitem__(self, index):
        subj = self.subj_ids[index]

        sample_id = np.random.randint(0, self.num_samples)
        rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_sub-%s_sample%d_rsfc.npy" % (subj, sample_id))
	#rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_%s_sample%d_rsfc.npy" % (subj, sample_id))
        subj_rsfc_data = np.load(rsfc_file)

        # All subject group contrast
        # group_contrast_path = os.path.join(self.group_contrast_dir, f'group_average_{self.task_id}.npy')
        # group_contrast = np.load(group_contrast_path)

        # N - 1 subject group contrast
        group_contrast_path = os.path.join(self.group_contrast_dir, f'group_contrast_without_{subj}.npy')
        group_contrast = np.load(group_contrast_path)[2*self.task_id:2*self.task_id+2]

        input_data = np.concatenate((subj_rsfc_data, group_contrast), axis=0)

        subj_task_data = np.load(os.path.join(self.contrast_dir, "%s_joint_LR_task_contrasts.npy" % subj))
        output_data = subj_task_data[2*self.task_id:2*self.task_id+2]

        # Steps to complete:
        # 1. Concatenate the group contrast with the rsfc data
        # 2. Select just the two hemis baseed on the task
        # Rest of the pipeline should be the same
        return torch.cuda.FloatTensor(input_data) , torch.cuda.FloatTensor(output_data)

    def __len__(self):
        return len(self.subj_ids)

class MultiGroupContrastMultipleSampleMeshDataset(Dataset):
    # Randomly samples an input and a group task contrast
    def __init__(self, subj_ids, rsfc_dir, contrast_dir, group_contrast_dir, task_list, num_samples=8):
        self.rsfc_dir = rsfc_dir
        self.contrast_dir = contrast_dir
        self.subj_ids = subj_ids
        self.num_samples = num_samples
        self.group_contrast_dir = group_contrast_dir
        self.task_list = task_list
        # self.task_samples = task_samples


    def __getitem__(self, index):
        subj = self.subj_ids[int(index / len(self.task_list))]

        sample_id = np.random.randint(0, self.num_samples)
        task_id = np.random.choice(self.task_list)

        rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_sub-%s_sample%d_rsfc.npy" % (subj, sample_id))
	    #rsfc_file = os.path.join(self.rsfc_dir, "joint_LR_%s_sample%d_rsfc.npy" % (subj, sample_id))
        subj_rsfc_data = np.load(rsfc_file)

        # N - 1 subject group contrast
        group_contrast_path = os.path.join(self.group_contrast_dir, f'normalized_group_contrast_without_{subj}.npy')
        group_contrast = np.load(group_contrast_path)[2*task_id:2*task_id+2]

        input_data = np.concatenate((subj_rsfc_data, group_contrast), axis=0)

        subj_task_data = np.load(os.path.join(self.contrast_dir, "%s_joint_LR_task_contrasts.npy" % subj))
        output_data = subj_task_data[2*task_id:2*task_id+2]

        # Steps to complete:
        # 1. Concatenate the group contrast with the rsfc data
        # 2. Select just the two hemis baseed on the task
        # Rest of the pipeline should be the same
        return torch.cuda.FloatTensor(input_data) , torch.cuda.FloatTensor(output_data)

    def __len__(self):
        return len(self.subj_ids) * len(self.task_list) #self.task_samples