import json 
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
from sklearn.model_selection import KFold

from tqdm import tqdm
import torch
import torch.nn.init as init
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models import *
import logging
logging.basicConfig(level=logging.INFO)

class MtrxType(Enum):
    ADJACENCY_MATRIX = 0
    EDGE_SEQUENCE = 1
    NODE_FEATURE_MATRIX = 2

def calculate_mean_loss_per_bin(true_values, predicted_values, num_bins=10):
    # Binning the true values
    bins = np.linspace(min(true_values), max(true_values), num_bins + 1)
    bin_indices = np.digitize(true_values, bins) - 1  # -1 to make bins 0-indexed

    mean_losses = []
    for i in range(num_bins):
        # Extract values in each bin
        indices = [index for index, bin_index in enumerate(bin_indices) if bin_index == i]
        bin_true_values = [true_values[index] for index in indices]
        bin_predicted_values = [predicted_values[index] for index in indices]
        
        # Calculate mean loss (mean absolute error) for each bin
        if bin_true_values:
            mean_loss = np.mean([abs(true - pred) for true, pred in zip(bin_true_values, bin_predicted_values)])
            mean_losses.append(mean_loss)
        else:
            mean_losses.append(0)

    return bins, mean_losses

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
        return load_npz(f'data/npzs/adjacency_mtrx/{index}.npz').todense()
    elif type == MtrxType.NODE_FEATURE_MATRIX:
        return load_npz(f'data/npzs/node_feature_mtrx/{index}.npz').todense()
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

def adjacency_to_edge_index(adjacency_matrix):
    src, dst = np.where(adjacency_matrix > 0)
    edge_index = np.stack((src, dst), axis=0)
    return torch.tensor(edge_index, dtype=torch.long)

def data_loading_all():
    # load data without splitting
    df_train = pd.read_csv('data/csvs/train.csv')
    difficulties = df_train['difficulty'].to_numpy()
    uuids = df_train['uuid'].to_numpy()
    df_climbs = pd.read_csv('data/csvs/climbs.csv')
    indices = np.arange(len(difficulties))
    valid_indices = []
    adjacency_matrices = []  # This will hold the adjacency matrices
    node_feature_matrices = []  # This will hold the node feature matrices
    edge_sequence_list = []  # This will hold the edge sequences
    valid_difficulties = []
    print("Loading data...")
    for i in indices:
        try: 
            node_feature_matrix = load_npz(f'data/npzs/node_feature_mtrx/{i}.npz').todense()
            node_feature_matrices.append(node_feature_matrix)
            adjacency_matrix = load_npz(f'data/npzs/adjacency_mtrx/{i}.npz').todense()
            adjacency_matrices.append(adjacency_matrix)
            edge_index = adjacency_to_edge_index(adjacency_matrix)
            edge_seq = torch.load(f'data/tensors/edge_sequence_{i}.pt')
            edge_sequence_list.append(edge_seq)
            x = torch.tensor(node_feature_matrix, dtype=torch.float)
            assert edge_seq.shape[0] == edge_index.shape[1], "Mismatch in number of edges vs edge sequence."
            assert x.shape[0] == adjacency_matrix.shape[0], "Mismatch in number of nodes vs features."
            assert edge_index.max() < x.shape[0], "edge_index contains node indices not in feature matrix."
            valid_indices.append(i)
            valid_difficulties.append(difficulties[i])
        except FileNotFoundError:
            pass
    train_indices, test_indices = train_test_split(
    valid_indices, test_size=0.2, random_state=1
    )
    index_mapping = {v: k for k, v in enumerate(valid_indices)}
    # valid_difficulties = (np.array(valid_difficulties) - np.mean(valid_difficulties))/ np.std(valid_difficulties)
    def generate_data_list(indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, difficulties):
        data_list = []
        for idx in indices:
            uuid = uuids[idx]
            idx = index_mapping[idx]
            edge_index = adjacency_to_edge_index(adjacency_matrices[idx])
            edge_attr = edge_sequence_list[idx]
            x = torch.tensor(node_feature_matrices[idx], dtype=torch.float)
            y = torch.tensor([difficulties[idx]], dtype=torch.float)
            data_object = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, uuid=uuid)
            data_list.append(data_object)
        return data_list
    train_data_list = generate_data_list(train_indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, valid_difficulties)
    test_data_list = generate_data_list(test_indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, valid_difficulties)
    return train_data_list, test_data_list
    
def data_loading(n_splits=5):
    df_train = pd.read_csv('data/csvs/train.csv')
    difficulties = df_train['difficulty'].to_numpy()
    indices = np.arange(len(difficulties))
    df_climbs = pd.read_csv('data/csvs/climbs.csv')
    uuids = df_train['uuid'].to_numpy()
    valid_indices = []
    adjacency_matrices = []  # This will hold the adjacency matrices
    node_feature_matrices = []  # This will hold the node feature matrices
    edge_sequence_list = []  # This will hold the edge sequences
    valid_difficulties = []
    valid_uuids = []
    print("Loading data...")
    for i in indices:
        try: 
            node_feature_matrix = load_npz(f'data/npzs/node_feature_mtrx/{i}.npz').todense()
            node_feature_matrices.append(node_feature_matrix)
            adjacency_matrix = load_npz(f'data/npzs/adjacency_mtrx/{i}.npz').todense()
            adjacency_matrices.append(adjacency_matrix)
            edge_index = adjacency_to_edge_index(adjacency_matrix)
            edge_seq = torch.load(f'data/tensors/edge_sequence_{i}.pt')
            edge_sequence_list.append(edge_seq)
            x = torch.tensor(node_feature_matrix, dtype=torch.float)
            ascentionist_count = df_climbs[df_climbs['uuid'] == uuids[i]]['ascensionist_count'].values[0]
            if ascentionist_count < 30:
                continue
            assert edge_seq.shape[0] == edge_index.shape[1], "Mismatch in number of edges vs edge sequence."
            assert x.shape[0] == adjacency_matrix.shape[0], "Mismatch in number of nodes vs features."
            assert edge_index.max() < x.shape[0], "edge_index contains node indices not in feature matrix."
            valid_indices.append(i)
            valid_difficulties.append(difficulties[i])  # Keep the valid difficulty
            valid_uuids.append(uuids[i])
        except FileNotFoundError:
            # difficulties = np.delete(difficulties, i)
            # indices = np.delete(indices, i)
            pass
    train_indices, test_indices = train_test_split(
        valid_indices, test_size=0.2, random_state=1
    )
    index_mapping = {v: k for k, v in enumerate(valid_indices)}
    # valid_difficulties = (np.array(valid_difficulties) - np.mean(valid_difficulties))/ np.std(valid_difficulties)
    # logging.info(f"Number of valid climbs: {len(valid_indices)}")
    def generate_data_list(indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, difficulties):
        data_list = []
        for idx in indices:
            idx = index_mapping[idx]
            edge_index = adjacency_to_edge_index(adjacency_matrices[idx])
            edge_attr = edge_sequence_list[idx]
            x = torch.tensor(node_feature_matrices[idx], dtype=torch.float)
            y = torch.tensor([difficulties[idx]], dtype=torch.float)
            data_object = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, uuid=valid_uuids[idx])
            data_list.append(data_object)
        return data_list
    
    def normalize_difficulties(data_list, mean, std):
        for data in data_list:
            data.y = (data.y - mean) / std
        return data_list
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    fold_data_lists = []
    fold_stds = []
    
    for train_idx, test_idx in kf.split(indices):
        train_indices = [indices[i] for i in train_idx if i in valid_indices]
        test_indices = [indices[i] for i in test_idx if i in valid_indices]
        train_difficulties = np.array([valid_difficulties[index_mapping[i]] for i in train_idx if i in valid_indices])
        mean_difficulty = np.mean(train_difficulties)
        std_difficulty = np.std(train_difficulties)
        fold_stds.append(std_difficulty)
        train_data_list = generate_data_list(train_indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, valid_difficulties)
        test_data_list = generate_data_list(test_indices, adjacency_matrices, node_feature_matrices, edge_sequence_list, valid_difficulties)
        train_data_list = normalize_difficulties(train_data_list, mean_difficulty, std_difficulty)
        test_data_list = normalize_difficulties(test_data_list, mean_difficulty, std_difficulty)
        fold_data_lists.append((train_data_list, test_data_list))

    return fold_data_lists, fold_stds

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

def run_training_and_evaluation(conv_types, k_values, train_loader, test_loader, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, lr=0.01, dropout_rate1=0.5, dropout_rate2=0.5, weight_decay=5e-4, num_epochs=200, mlflow_run=None, counter=None):
    input_dim = next(iter(train_loader)).x.shape[1]
    model = SimpleGNN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, conv_types, k_values, dropout_rate1, dropout_rate2)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    mse_loss_criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
    l1_loss_criterion = torch.nn.L1Loss()    # Mean Absolute Error for validation
    best_val_loss = float('inf')
    val_loss_list = []
    train_loss_list = []
    true_values = []
    predicted_values = []

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        total_train_loss = 0.0
        model.train()

        # Training phase
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch).squeeze(-1)
            loss = mse_loss_criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_loss_list.append(avg_train_loss)

        total_val_loss = 0.0
        model.eval()
        prediction_details = []
        # Validation phase
        for batch in test_loader:
            with torch.no_grad():
                out = model(batch).squeeze(-1)
                loss = l1_loss_criterion(out, batch.y)
                total_val_loss += loss.item() * batch.num_graphs
                true_values.extend(batch.y.tolist())
                predicted_values.extend(out.tolist())
                batch_uuids = batch.uuid

                for actual, predicted, uuid in zip(batch.y.tolist(), out.tolist(), batch_uuids):
                    prediction_details.append({
                        'uuid': uuid,
                        'actual': actual,
                        'predicted': predicted,
                        'loss': abs(actual - predicted)
                    })
        avg_val_loss = total_val_loss / len(test_loader.dataset)
        val_loss_list.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
#            torch.save(model.state_dict(), 'best_model.pt')
        if mlflow_run:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    bins, mean_losses = calculate_mean_loss_per_bin(true_values, predicted_values)
    # Plotting True Values vs Predicted Values
    plt.figure(figsize=(10, 5))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(f"data/pngs/true_vs_pred_{counter}.png")
    # Plotting Mean Loss by Bins
    plt.figure(figsize=(10, 5))
    plt.bar(bins[:-1], mean_losses, width=np.diff(bins), align="edge", edgecolor='black')
    plt.title('Mean Loss by Bins of True Values')
    plt.xlabel('Bins of True Values')
    plt.ylabel('Mean Loss')
    plt.grid(True)
    plt.savefig(f"data/pngs/mean_loss_by_bins_{counter}.png")
    plt.close('all')
    
    if mlflow_run: 
        mlflow.log_param("lr", lr)
        mlflow.log_param("dropout_rate1", dropout_rate1)
        mlflow.log_param("dropout_rate2", dropout_rate2)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("hidden_dim1", hidden_dim1)
        mlflow.log_param("hidden_dim2", hidden_dim2)
        mlflow.log_param("hidden_dim3", hidden_dim3)
        mlflow.log_artifact(f"data/pngs/mean_loss_by_bins_{counter}.png")
        mlflow.log_artifact(f"data/pngs/true_vs_pred_{counter}.png")
        top_10_worst_predictions = sorted(prediction_details, key=lambda x: x['loss'], reverse=True)[:10]

        # Load climb names
        df_climbs = pd.read_csv('data/csvs/climbs.csv')

        # Add climb names to the predictions
        for pred in top_10_worst_predictions:
            climb_name = df_climbs[df_climbs['uuid'] == pred['uuid']]['name'].values[0]
            pred['climb_name'] = climb_name
            logging.info(f"{climb_name}, Actual: {pred['actual']}, Predicted: {pred['predicted']}, Loss: {pred['loss']}")

        # Write to a JSON file
        with open(f"data/jsons/top_10_worst_predictions_{counter}.json", "w") as file:
            json.dump(top_10_worst_predictions, file, indent=4)

        # Log the JSON file with MLflow
        mlflow.log_artifact(f"data/jsons/top_10_worst_predictions_{counter}.json")
    return best_val_loss

if __name__ == "__main__":
    fold_val_losses = []
    # train_data, test_data = data_loading()

    mlflow.set_experiment("Tuned Model")
    fold_data_lists, fold_stds = data_loading()
    fold_val_losses = []
    hidden_dim1 = 256
    hidden_dim2 = 512   
    hidden_dim3 = 256
    lr = 0.0004761947456361084
    dropout_rate1 = 0.4749033543430395
    dropout_rate2 = 0.3987789183651262
    weight_decay = 1.8746053259016402e-05
    conv_types = [ChebConv, ChebConv, TransformerConv]
    k_values = [2, 2, 2]
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    for counter, (train_data, test_data) in enumerate(fold_data_lists):
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        # Run the training and validation process for each fold
        best_val_loss = run_training_and_evaluation(
        conv_types, k_values, train_loader, test_loader,
        hidden_dim1, hidden_dim2, hidden_dim3,
        lr, dropout_rate1, dropout_rate2, weight_decay, num_epochs=100, mlflow_run=True, counter=counter)
        fold_val_losses.append(best_val_loss)
    # mlflow.log_metric("fold_val_losses", fold_val_losses*np.array(fold_stds))
    mlflow.log_metric("mean_val_loss", np.mean(fold_val_losses*np.array(fold_stds)))

