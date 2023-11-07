import mlflow
import networkx as nx
from scipy.sparse import csr_matrix
import ast
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from enum import Enum

from tqdm import tqdm
import torch
import torch.nn.init as init
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from models import *

class MtrxType(Enum):
    ADJACENCY_MATRIX = 0
    EDGE_SEQUENCE = 1
    NODE_FEATURE_MATRIX = 2

def generate_graphs(use_features=False):
    # List to store generated graphs
    graphs = []

    df_train, df_nodes = pd.read_csv('data/csvs/train.csv'), pd.read_csv('data/csvs/nodes.csv')

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

def graph_preprocessing():
    df_train, df_nodes = pd.read_csv('data/csvs/train.csv'), pd.read_csv('data/csvs/nodes.csv')

    # Normalize the screw_angle and x, y coordinates
    df_nodes['norm_screw_angle'] = df_nodes['screw_angle'] / 360
    df_nodes['norm_x'] = df_nodes['x'] / df_nodes['x'].max()
    df_nodes['norm_y'] = df_nodes['y'] / df_nodes['y'].max()

    # Create dummy variables for SKU and hold type
    df_nodes = pd.get_dummies(df_nodes, columns=['sku', 'hold_type'])
    df_nodes = df_nodes.drop(columns=['name', 'screw_angle', 'x', 'y'])

    # Create interaction terms for norm_screw_angle and SKU dummies
    for sku_col in [col for col in df_nodes.columns if col.startswith('sku_')]:
        df_nodes[f'{sku_col}_angle_interaction'] = df_nodes[sku_col] * df_nodes['norm_screw_angle']

    for index, row in tqdm(df_train.iterrows()):
        nodes_list = ast.literal_eval(row['nodes'])
        hold_types = ast.literal_eval(row['hold_type'])

        sparse_matrix = build_sparse_matrix(df_nodes, nodes_list, hold_types)
        save_npz(f'data/npzs/node_feature_mtrx/{index}.npz', sparse_matrix)
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
    elif type == MtrxType.EDGE_SEQUENCE:
        return torch.load(f'data/tensors/edge_sequence_{index}.pt')
    # elif type == MtrxType.MIRRORED_ADJACENCY_MATRIX:
    #     return load_npz(f'data/npzs/adjacency_mtrx/{index}_mirrored.npz')
    else:
        raise Exception('Invalid type')

def sparse_to_torch_tensor(sparse_matrix):
    # Convert a scipy sparse matrix to a torch edge_index tensor
    coo_matrix = sparse_matrix.tocoo()
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    edge_index = torch.tensor(indices, dtype=torch.long)
    return edge_index
def data_loading():
    df_train = pd.read_csv('data/csvs/train.csv')
    difficulties = df_train['difficulty'].to_numpy()[:1000]
    indices = np.arange(len(difficulties))
    adjacency_matrices = []  # This will hold the adjacency matrices
    node_feature_matrices = []  # This will hold the node feature matrices
    edge_sequence_list = []  # This will hold the edge sequences
    print("Loading data...")
    for i in indices:
        try:
            edge_sequence_tensor = torch.load(f"data/tensors/edge_sequence_{i}.pt")
            edge_sequence_list.append(edge_sequence_tensor)
            adjacency_matrices.append(load_mtrx(i, MtrxType.ADJACENCY_MATRIX))
            node_feature_matrices.append(load_mtrx(i, MtrxType.NODE_FEATURE_MATRIX))
            # Load edge sequence tensors
        except FileNotFoundError as e:
            print(f"Data for index {i} not found: {e}")
            difficulties = np.delete(difficulties, np.argwhere(indices == i))
            indices = np.delete(indices, np.argwhere(indices == i))
            continue
    # Splitting the data
    # print(difficulties)
    train_indices, test_indices = train_test_split(
        range(len(indices)), test_size=0.2, random_state=1
    )
    # print(indices)

    def generate_data_list(indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, difficulties):
        data_list = []
        for idx in indices:
            edge_index = adjacency_to_edge_index(adjacency_matrices[idx].todense())
            edge_attr = edge_sequence_list[idx]
            x = torch.tensor(node_feature_matrices[idx].todense(), dtype=torch.float)
            y = torch.tensor([difficulties[idx]], dtype=torch.float)
            data_object = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data_object)
        return data_list
    train_data_list = generate_data_list(train_indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, difficulties)
    test_data_list = generate_data_list(test_indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, difficulties)
    return train_data_list, test_data_list

def adjacency_to_edge_index(adjacency_matrix):
    src, dst = np.where(adjacency_matrix > 0)
    edge_index = np.stack((src, dst), axis=0)
    return torch.tensor(edge_index, dtype=torch.long)

def compare_edge_sequence_with_edge_index(edge_sequence_tensor, edge_index):
    edge_mapping = {}
    for i, (src, dst) in enumerate(edge_sequence_tensor):
        # Create a mapping from each edge to its position in the sequence
        edge_mapping[(src.item(), dst.item())] = i
    
    # Check if each edge in edge_index is in the edge sequence
    for i, (src, dst) in enumerate(edge_index.t()):
        edge = (src.item(), dst.item())
        if edge in edge_mapping:
            print(f"Edge {edge} from edge_index is at position {edge_mapping[edge]} in the edge sequence.")
        else:
            print(f"Edge {edge} from edge_index is not in the edge sequence.")

def convert_to_categorical(difficulties, min_difficulty, max_difficulty, num_classes):
    difficulty_range = np.linspace(min_difficulty, max_difficulty, num_classes + 1)
    categories = np.digitize(difficulties, bins=difficulty_range, right=True) 
    categories = np.where(categories == num_classes, num_classes - 1, categories)
    return categories

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

def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data).squeeze(-1)  
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(data, model, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        predictions = predictions.squeeze(-1)  
        loss = criterion(predictions, data.y)
    return loss.item()

def run_training_and_evaluation(num_epochs=100, lr=0.01, save_path='best_model.pt'):
    mlflow.set_experiment("GNN_Training_Results")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", lr) 
        # Split data into training and testing sets
        train_data_list, test_data_list = data_loading()

        # Get the input dimension from the node features of the first Data object in the train_data_list
        input_dim = train_data_list[0].x.size(1)

        # Initialize model, optimizer, and loss function
        hidden_dim1 = 256
        hidden_dim2 = 128
        mlflow.log_param("hidden_dim1", hidden_dim1)
        mlflow.log_param("hidden_dim2", hidden_dim2)
        model = SimpleGNN(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)
        optimizer = torch.optim.Adam(model.parameters(), lr)
        criterion = torch.nn.L1Loss()   # Mean Squared Error for regression

        best_val_loss = float('inf')
        val_loss_list = []
        train_loss_list = []
        print("Training...")

        for epoch in tqdm(range(num_epochs)):
            total_train_loss = 0.0
            model.train()
            for data in train_data_list:  # Directly iterate over pre-processed data
                optimizer.zero_grad()
                out = model(data).squeeze(-1)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_data_list)
            total_val_loss = 0.0
            model.eval()
            with torch.no_grad():  # No need to compute gradients for validation data
                for data in test_data_list:
                    out = model(data).squeeze(-1)
                    loss = criterion(out, data.y)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(test_data_list)
            print(f'Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            # Log metrics for this epoch with MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            # Saving the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with validation loss: {best_val_loss:.4f}")

            # Append statistics for later analysis
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)
        mlflow.end_run()

        
    model.load_state_dict(torch.load(save_path))
    return model, val_loss_list, train_loss_list


if __name__ == "__main__":
    # graph_preprocessing()
    model, val_loss, train_loss = run_training_and_evaluation()
