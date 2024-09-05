import os
import numpy as np

import torch
from torch.nn.parameter import Parameter

from model.brain_surf_gnn import BrainSurfGCN
from utils.parser import test_args
from utils.dataset import MultiContrastGraphDataset
from utils.utilities import CONTRASTS, save_checkpoint
from torch_geometric.data import Data
from torch_geometric.transforms.face_to_edge import FaceToEdge


if __name__ == "__main__":

    args = test_args()

    """Init"""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpus

    output_dir = os.path.join(args.save_dir, args.ver)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise Exception("Output dir exists: %s" % output_dir)


    """Load Data"""
    subj_ids = np.genfromtxt(args.subj_list, dtype="<U13")

    model = BrainSurfGCN(
        in_ch=args.n_channels_per_hemi*2 + 3, # 3 for the XYZ of the vertices
        out_ch=args.n_output_channels*2,
        hidden_channels=[64, 64, 128, 128])
    model.cuda()

    state_dict = torch.load(args.checkpoint_file)['state_dict']
    model_state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        if name not in model_state_dict:
            continue
        if 'none' in name:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state_dict[name].copy_(param)
    
    model.eval()

    mesh = np.load(args.mesh_dir, allow_pickle=True)
    f2e = FaceToEdge()

    with torch.no_grad():
        for i in range(len(subj_ids)):
            print(i+1, "/", len(subj_ids))
            subj = subj_ids[i]
            pred_file = os.path.join(output_dir, "%s_pred.npy" % subj)
            if not os.path.exists(pred_file):
                subj_pred = []
                for sample_id in range(args.n_samples_per_subj):
                    rsfc_file = os.path.join(args.rsfc_dir, "joint_LR_sub-%s_sample%d_rsfc.npy" % (subj, sample_id))
                    subj_rsfc_data = np.load(rsfc_file).T # shape = V x 2*ICs

                    data = Data()
                    data.face = torch.tensor(mesh['F'].T)
                    data.num_nodes = mesh['V'].shape[0]
                    data = f2e(data)

                    # Adds positional awareness
                    input_data = np.concatenate([subj_rsfc_data, mesh['V']], axis=-1) # shape = V x 2*ICs + 3

                    data.x = torch.FloatTensor(input_data)
                    data.ptr = torch.tensor([0, 1]) # mimics the batch system
                    sample_pred = model(data.cuda())
                    
                    
                    subj_pred.append(sample_pred.cpu().detach().numpy().squeeze(0))
                subj_pred = np.asarray(subj_pred)
                np.save(pred_file, subj_pred)

print("Finished prediction")
