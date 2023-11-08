import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnn import adjacency_to_edge_index

for i in range(1000):
    try: 
        node_feature_matrix = load_npz(f'data/npzs/node_feature_mtrx/{i}.npz').todense()
        adjacency_matrix = load_npz(f'data/npzs/adjacency_mtrx/{i}.npz').todense()
        edge_index = adjacency_to_edge_index(adjacency_matrix)
        x = torch.tensor(node_feature_matrix, dtype=torch.float)
        print(x.shape, edge_index, edge_index.max(), x.shape[0])
        assert x.shape[0] == adjacency_matrix.shape[0], "Mismatch in number of nodes vs features."
        assert edge_index.max() < x.shape[0], "edge_index contains node indices not in feature matrix."

    except FileNotFoundError:
        pass
