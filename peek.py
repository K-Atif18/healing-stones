import torch
from torch_geometric.data import Data
from torch.serialization import safe_globals

with safe_globals([Data]):
    graphs = torch.load("dataset_graphs.pt", weights_only=False)

print(graphs)
