import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class BroadwayGNN(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        for _ in range(3):
            self.convs.append(
                HeteroConv({
                    ("person", "acted_in", "show"):
                        GATConv((-1, -1), 128),
                    ("person", "produced", "show"):
                        GATConv((-1, -1), 128),
                    ("show", "at", "theater"):
                        GATConv((-1, -1), 128),
                }, aggr="sum")
            )

        self.show_lin = Linear(128, 128)
        self.out = torch.nn.Sequential(
            Linear(128, 64),
            torch.nn.ReLU(),
            Linear(64, 1)
        )

    def forward(self, data):
        x_dict = data.x_dict

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        show_emb = self.show_lin(x_dict["show"])
        return self.out(show_emb).squeeze(-1)
