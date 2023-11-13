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

for i in range(4211):
    try: 
        print(i)
        node_feature_matrix = load_npz(f'data/npzs/node_feature_mtrx/{i}.npz').todense()
        adjacency_matrix = load_npz(f'data/npzs/adjacency_mtrx/{i}.npz').todense()
        edge_index = adjacency_to_edge_index(adjacency_matrix)
        edge_seq = torch.load(f'data/tensors/edge_sequence_{i}.pt')
        x = torch.tensor(node_feature_matrix, dtype=torch.float)
        assert edge_seq.shape[0] == edge_index.shape[1], f"Mismatch in number of edges vs edge sequence. index {i}, edge_seq.shape[0] = {edge_seq.shape[0]}, edge_index.shape[1] = {edge_index.shape[1]}."
        assert x.shape[0] == adjacency_matrix.shape[0], "Mismatch in number of nodes vs features."
        assert edge_index.max() < x.shape[0], "edge_index contains node indices not in feature matrix."

    except FileNotFoundError:
        # difficulties = np.delete(difficulties, i)
        # indices = np.delete(indices, i)
            pass
