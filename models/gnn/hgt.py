import torch
from torch_geometric.nn import HGTConv

class HGTEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, data):
        super().__init__()

        node_types, edge_types = data.metadata()

        self.lin = torch.nn.ModuleDict()
        for ntype in node_types:
            in_dim = data[ntype].x.shape[1]
            self.lin[ntype] = torch.nn.Linear(in_dim, hidden_dim)

        self.conv1 = HGTConv(hidden_dim, hidden_dim, data.metadata(), heads=2)
        self.conv2 = HGTConv(hidden_dim, hidden_dim, data.metadata(), heads=2)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin[k](v) for k, v in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict
