import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import sklearn

from model.brain_surf_gnn import BrainSurfGCN
from utils import experiment
from utils.parser import train_args
from utils.dataset import MultipleSampleMeshDataset
from utils.utilities import CONTRASTS, parse_contrasts_names, save_checkpoint, contrast_mse_loss

if __name__ == "__main__":

    # args = train_args()

    # #output_name = "%s_feat%d_s%d_c%d_lr%s_seed%d" % (args.ver, args.n_feat_channels, args.n_samples_per_subj, args.n_channels_per_hemi, str(args.lr), args.seed)
    # #output_dir = os.path.join(args.save_dir, output_name)
    # #if not os.path.exists(output_dir):
    # #    os.makedirs(output_dir)
    # #else:
    # #    raise Exception("Output dir exists: %s" % output_dir)
    # #writer = SummaryWriter(os.path.join(output_dir, "logs"))

    
    # """Init model"""
    # """two hemispheres are concatenated"""
    n_channels_per_hemi = 15
    n_output_channels = 47
    model = BrainSurfGCN(
        in_ch=n_channels_per_hemi*2 + 3,
        out_ch=n_output_channels*2)
    model.cuda()

    # # print(torch.cuda.memory_summary())

    print(model)

    # print(summary(model, (n_channels_per_hemi*2, 32492)))
