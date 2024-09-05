import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BDLayer(nn.Module):
    def __init__(self, input_features, output_features, aggr='mean', activation=nn.LeakyReLU):
        super(BDLayer, self).__init__()
        self.conv = GCNConv(input_features, output_features, aggr=aggr)
        self.activation = activation()
        self.bn = nn.BatchNorm1d(output_features)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        x = self.bn(x)
        return x

class BrainSurfGCN(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_channels=[32, 32, 64, 64]):
        super(BrainSurfGCN, self).__init__()
        assert(len(hidden_channels) > 0)
        # self.in_conv = GCNConv(input_features, hidden_channels[0], aggr='mean')
        down_layers = []
        for i in range(len(hidden_channels)):
            if i == 0:
                down_layers.append(BDLayer(in_ch, hidden_channels[i], aggr='mean'))
            else:
                down_layers.append(BDLayer(hidden_channels[i-1], hidden_channels[i], aggr='mean'))
        self.down_layers = nn.Sequential(*down_layers)

        init_up = hidden_channels[-1]
        up_layers = []
        up_layers.append(BDLayer(init_up, hidden_channels[-1], aggr='mean'))
        for i in range(1, len(hidden_channels)):
            up_layers.append(BDLayer(hidden_channels[-i], hidden_channels[-i-1], aggr='mean'))
        self.up_layers = nn.Sequential(*up_layers)

        self.lin = Linear(hidden_channels[0], out_ch, bias=True)

        self.in_ch = in_ch
        self.out_ch = out_ch
    
    def forward(self, data):
        # data must be a torch_geometric Data() object
        
        batch = len(data.ptr) - 1
        num_nodes = data.num_nodes // batch # there are batch * num_nodes in the data structure, makes one big bipartite graph for batched data
        x = data.x
        edge_index = data.edge_index
        res = []
        for dl in self.down_layers:
            x = dl(x, edge_index)
            res.append(x)

        for i, ul in enumerate(self.up_layers):
            x = res[-(i+1)] + ul(x, edge_index)

        x = self.lin(x)
        x = x.reshape(batch, num_nodes, self.out_ch)
        return x.transpose(1,2)