import torch
from torch import nn
from layers import GraphConv, GraphAttConv


class ExitBlock(nn.Module):

    def __init__(self, nhid, nclass, dropout=0.06, exit_type=None):
        nhid1 = nhid
        super(ExitBlock, self).__init__()
        self.dropout1 = nn.Dropout(p=0.4)
        self.out_layer =GraphConv (nhid, nhid)
        self.fc = torch.nn.Linear(nhid, nclass)
        self.exit_layer = nn.ModuleList()
        self.get_exit_block()

    def get_exit_block(self):
        self.exit_layer.append(self.dropout1)
        self.exit_layer.append(self.out_layer)
        self.exit_layer.append(self.fc)

    def forward(self, X):
        x, adj = X[0], X[1]
        for i, layer in enumerate(self.exit_layer):
            if isinstance(layer, GraphConv) or isinstance(layer, GraphAttConv):
                x = layer(x, adj)
            else:
                x = layer(x)
        return x
