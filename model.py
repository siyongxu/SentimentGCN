import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, adj, out_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.adj = adj
        self.out_dim = out_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions

        self.optimizer = torch.train.AdamOptimizer(learning_rate=0.02)

        self.conv1 = GCNConv(adj.shape[0], 200)
        self.conv2 = GCNConv(200, out_dim)


    def forward(self, input):
        x = F.relu(self.conv1(input))
        return F.softmax(self.conv2(x))



