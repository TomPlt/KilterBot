import networkx as nx
from scipy.sparse import csr_matrix
import ast
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
from enum import Enum

from tqdm import tqdm
import torch
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class MtrxType(Enum):
    ADJACENCY_MATRIX = 0
    NODE_FEATURE_MATRIX = 1

def generate_graphs(use_features=False):
    # List to store generated graphs
    graphs = []

    df_train, df_nodes = data_load_in()

    for index, row in df_train.iterrows():
        G = nx.Graph()  # Initialize a new graph for this row
        
        coordinates = ast.literal_eval(row['coordinates'])
        nodes = ast.literal_eval(row['nodes'])
        if use_features:
            hold_variants = ast.literal_eval(row['hold_type'])

        # Add the node_ids to the graph with or without features
        for i, node_id in enumerate(nodes):
            if use_features:
                node_features = df_nodes.loc[node_id].to_dict()
                G.add_node(node_id, 
                        coordinates=coordinates[i], 
                        hold_variant=hold_variants[i], 
                        **node_features)
            else:
                G.add_node(node_id)

        # Calculate distances and create edges:
        for idx, coord in enumerate(coordinates):
            distances = {}
            for target_idx, target_coord in enumerate(coordinates):
                if idx != target_idx:
                    dist = np.linalg.norm(np.array(coord) - np.array(target_coord))
                    distances[target_idx] = dist
            # Get two nearest neighbors
            nearest_neighbors = sorted(distances.keys(), key=lambda x: distances[x])[:2]
            for neighbor_idx in nearest_neighbors:
                G.add_edge(nodes[idx], nodes[neighbor_idx])
        graphs.append(G)
    return graphs

def build_sparse_matrix(df_nodes, nodes_list, hold_types):
    # Create a deep copy of df_nodes to modify
    temp_df = df_nodes.copy()
    
    # Populate the 'hold_type' column with the new hold_type for this climb
    for node, hold_type in zip(nodes_list, hold_types):
        temp_df.at[node, 'hold_type'] = hold_type
    
    # One-hot encode the 'hold_type' column
    temp_df = pd.get_dummies(temp_df, columns=['hold_type'])
    
    # Ensure all hold_type_columns exist in the DataFrame
    hold_type_columns = [
        'hold_type_Start',
        'hold_type_Middle', 
        'hold_type_Finish', 
        'hold_type_Foot Only', 
    ]
    for col in hold_type_columns:
        if col not in temp_df.columns:
            temp_df[col] = 0

    # Reorder the columns to ensure the same sequence
    columns_order = [col for col in temp_df.columns if col not in hold_type_columns] + hold_type_columns
    temp_df = temp_df.reindex(columns=columns_order)
    # Build matrix
    matrix = np.full(temp_df.shape, np.nan)
    matrix[nodes_list] = temp_df.loc[nodes_list].values
    matrix[np.isnan(matrix)] = 0
    return csr_matrix(matrix)

def visualize_graph(graphs: list, index):
    G = graphs[index]

    pos = nx.get_node_attributes(G, 'coordinates')

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=15, font_weight='bold')
    for node, attrs in G.nodes(data=True):
        # Creating a string of features for demonstration. 
        # Adjust accordingly to your use case.
        s = f"hold_variant: {attrs['hold_variant']}\n" + \
            "\n".join([f"{key}: {value}" for key, value in attrs.items() if key not in ['coordinates', 'hold_variant']])
        
        plt.annotate(s, (pos[node][0],pos[node][1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

    plt.title('Graph Visualization with Features')
    plt.show()

def graph_preprocessing():
    df_train, df_nodes = pd.read_csv('data/csvs/train.csv'), pd.read_csv('data/csvs/nodes.csv')
    df_nodes = pd.get_dummies(df_nodes, columns=['sku', 'hold_type'])
    df_nodes = df_nodes.drop(columns=['name'])
    df_nodes.screw_angle /= 360
    df_nodes.x /= df_nodes.x.max()
    df_nodes.y /= df_nodes.y.max()
    hold_type_columns = [
    'hold_type_Foot Only', 
    'hold_type_Middle', 
    'hold_type_Finish', 
    'hold_type_Start'
    ]
    for index, row in tqdm(df_train.iterrows()):
        nodes_list = ast.literal_eval(row['nodes'])
        hold_types = ast.literal_eval(row['hold_type'])
        
        sparse_matrix = build_sparse_matrix(df_nodes, nodes_list, hold_types)
        save_npz(f'data/npzs/node_feature_mtrx/{index}.npz', sparse_matrix)
        # print(climb_sparse_matrices[index].isna().sum())
# print(df_nodes.head(), df_nodes.screw_angle.max())

def create_adjacency_matrix(edges, dim):
    adjacency_matrix = np.zeros((dim, dim), dtype=np.int8)
    # For now undirected 
    for edge in edges:
        adjacency_matrix[edge[0]][edge[1]] = 1
        adjacency_matrix[edge[1]][edge[0]] = 1 
    return csr_matrix(adjacency_matrix)

def load_mtrx(index, type:MtrxType):
    if type == MtrxType.ADJACENCY_MATRIX:
        return load_npz(f'data/npzs/adjacency_mtrx/{index}.npz')
    elif type == MtrxType.NODE_FEATURE_MATRIX:
        return load_npz(f'data/npzs/node_feature_mtrx/{index}.npz')
    else:
        raise Exception('Invalid type')

def sparse_to_torch_tensor(sparse_matrix):
    # Convert a scipy sparse matrix to a torch edge_index tensor
    coo_matrix = sparse_matrix.tocoo()
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    edge_index = torch.tensor(indices, dtype=torch.long)
    return edge_index

def data_loadin():
    df_train, df_nodes = pd.read_csv('data/csvs/train.csv'), pd.read_csv('data/csvs/nodes.csv')
    difficulties = df_train.difficulty.to_numpy()
    adjacency_matrices = []
    node_feature_matrices = []
    print("Loading matrices...")
    for i in tqdm(range(len(difficulties))):
        adjacency_matrices.append(load_mtrx(i, MtrxType.ADJACENCY_MATRIX))
        node_feature_matrices.append(load_mtrx(i, MtrxType.NODE_FEATURE_MATRIX))
    return adjacency_matrices, node_feature_matrices, difficulties

def adjacency_to_edge_index(adjacency_matrix):
    src, dst = np.where(adjacency_matrix > 0)
    edge_index = np.stack((src, dst), axis=0)
    return torch.tensor(edge_index, dtype=torch.long)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)  # First convolutional layer
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)  # New second convolutional layer
        self.conv3 = GCNConv(hidden_dim2, output_dim)  # Third convolutional layer, previously the second
        self.lin = torch.nn.Linear(output_dim, 1)  # Linear layer remains unchanged

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply first convolutional layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        # Apply new second convolutional layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        # Apply third convolutional layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)

        # Global mean pooling and linear layer
        x = global_mean_pool(x, data.batch)  # Ensure 'data.batch' is provided
        x = self.lin(x)

        return x


def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()



def run_training(adjacency_matrices, node_feature_matrices, difficulties, num_epochs=100, save_path='best_model.pt'):
    model = SimpleGNN(input_dim=node_feature_matrices[0].shape[1], hidden_dim1=128, hidden_dim2=32, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()  

    best_loss = float('inf')  # Initialize best loss as infinity
    print("Training...")
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for i in range(len(difficulties)):
            # Converting sparse matrix to edge_index
            edge_index = adjacency_to_edge_index(adjacency_matrices[i].todense())
            x = torch.tensor(node_feature_matrices[i].todense(), dtype=torch.float)
            y = torch.tensor([difficulties[i]], dtype=torch.float)  
            data = Data(x=x, edge_index=edge_index, y=y)
            
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(difficulties)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')

        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with loss: {best_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model

def run_inference(model, adjacency_matrices, node_feature_matrices, difficulties):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()  

    for i in range(len(difficulties)):
        # Converting sparse matrix to edge_index
        edge_index = adjacency_to_edge_index(adjacency_matrices[i].todense())
        x = torch.tensor(node_feature_matrices[i].todense(), dtype=torch.float)
        y = torch.tensor([difficulties[i]], dtype=torch.float)  
        data = Data(x=x, edge_index=edge_index, y=y)
        out = model(data)
        loss = criterion(out, data.y.unsqueeze(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(difficulties)
    print(f'Loss: {avg_loss:.4f}')




if __name__ == "__main__":
    adjacency_matrices, node_feature_matrices, difficulties = data_loadin()
    run_training(adjacency_matrices, node_feature_matrices, difficulties)
    # print(difficulties[:10
    # ])
    model = SimpleGNN(input_dim=node_feature_matrices[0].shape[1], hidden_dim1=128, hidden_dim2=32, output_dim=1)
    model.load_state_dict(torch.load("best_model.pt"))
    run_inference(model, adjacency_matrices, node_feature_matrices, difficulties)
    # model.eval()