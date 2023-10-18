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

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):  # 'num_classes' is the number of classes you're predicting
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)  # First convolutional layer
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)  # Second convolutional layer
        self.lin = torch.nn.Linear(hidden_dim2, num_classes)  # A linear layer to get to the correct number of classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)  # Dropout for regularization
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Global pooling - aggregates node features into graph features
        x = global_mean_pool(x, data.batch)  # If using DataLoader, data.batch indicates to which graph a node belongs
        # Final linear layer - it uses the graph features to make a prediction
        x = self.lin(x)  # No softmax here as we output raw logits
        return x

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

def adjusted_accuracy(true_labels, predicted_labels):
    off_by_one = 0
    exact = 0
    total = len(true_labels)

    for true, predicted in zip(true_labels, predicted_labels):
        if true == predicted:
            exact += 1
        elif abs(true - predicted) == 1:  # off by one level
            off_by_one += 1

    # you can adjust the scoring as you see fit here
    score = (exact + 0.5 * off_by_one) / total  # giving half credit for off-by-one predictions
    return score

def ordinal_loss(outputs, targets, num_classes):
    # Expand targets for all classes
    expanded_targets = targets.view(-1, 1).repeat(1, num_classes)

    # Create a class matrix
    class_matrix = torch.arange(num_classes, device=outputs.device).view(1, -1).repeat(targets.size(0), 1)

    # Create ordinal targets matrix
    ordinal_targets = (class_matrix <= expanded_targets).float()

    # Calculate log of probabilities
    log_probs = F.log_softmax(outputs, dim=1)

    # Calculate probabilities (you omitted this step)
    probs = torch.exp(log_probs)  # or you can use F.softmax(outputs, dim=1) which was omitted in your code
    ordinal_ce_loss = -torch.mean(ordinal_targets * log_probs + (1 - ordinal_targets) * torch.log(1 - probs.clamp(min=1e-5)))

    return ordinal_ce_loss


def run_training_and_evaluation(adjacency_matrices, node_feature_matrices, difficulties, num_epochs=20, save_path='best_model.pt'):
    mlflow.set_experiment("GNN_Training_Results")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", 0.01) 
        # Split data into training and testing sets
        num_classes = 20  # for example, if you have 20 classes
        difficulties = convert_to_categorical(difficulties, min_difficulty=10.0, max_difficulty=29.0, num_classes=num_classes)
        train_adj_matrices, test_adj_matrices, train_node_features, test_node_features, train_difficulties, test_difficulties = train_test_split(
            adjacency_matrices, node_feature_matrices, difficulties, test_size=0.1, random_state=1, stratify=difficulties)
        # Initialize model, optimizer, and loss function
        model = SimpleGNN(input_dim=node_feature_matrices[0].shape[1], hidden_dim1=128, hidden_dim2=32, num_classes=num_classes)  # Ensure correct 'num_classes'
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()  

        best_val_loss = float('inf')
        val_loss_list = []
        train_loss_list = []
        accuracy_list = []
        print("Training...")

        for epoch in tqdm(range(num_epochs)):
                total_train_loss = 0.0
                correct_train_predictions = 0
                predicted_labels = []
                true_labels = []
                model.train()
                for i in range(len(train_difficulties)):
                    edge_index = adjacency_to_edge_index(train_adj_matrices[i].todense())
                    x = torch.tensor(train_node_features[i].todense(), dtype=torch.float)
                    y = torch.tensor([train_difficulties[i]], dtype=torch.long)
                    data = Data(x=x, edge_index=edge_index, y=y)

                    optimizer.zero_grad()
                    out = model(data)  
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    _, predicted = torch.max(out, 1)  
                    correct_train_predictions += (predicted == y).sum().item()  # Compare with ground truth

                avg_train_loss = total_train_loss / len(train_difficulties)
                train_accuracy = correct_train_predictions / len(train_difficulties)  # Calculate training accuracy
                total_val_loss = 0.0
                correct_val_predictions = 0
                model.eval()
                for i in range(len(test_difficulties)):
                    edge_index = adjacency_to_edge_index(test_adj_matrices[i].todense())
                    x = torch.tensor(test_node_features[i].todense(), dtype=torch.float)
                    y = torch.tensor([test_difficulties[i]], dtype=torch.long)
                    data = Data(x=x, edge_index=edge_index, y=y)
                    out = model(data)  
                    loss = criterion(out, y)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(out, 1)
                    predicted_labels.append(predicted.item())
                    true_labels.append(test_difficulties[i])
                avg_val_loss = total_val_loss / len(test_difficulties)
                val_accuracy = correct_val_predictions / len(test_difficulties)
                adj_accuracy = adjusted_accuracy(true_labels, predicted_labels)

                print(f'Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {adj_accuracy:.4f}')
                # Log metrics for this epoch with MLflow
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                # Saving the best model based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), save_path)
                    print(f"Best model saved with validation loss: {best_val_loss:.4f}")

                # Append statistics for later analysis
                train_loss_list.append(avg_train_loss)
                val_loss_list.append(avg_val_loss)
                accuracy_list.append(val_accuracy)  # Track validation accuracy per epoch
        mlflow.end_run()
        
    model.load_state_dict(torch.load(save_path))
    return model, val_loss_list, train_loss_list


if __name__ == "__main__":
    adjacency_matrices, node_feature_matrices, difficulties = data_loadin()
    model, val_loss, train_loss = run_training_and_evaluation(adjacency_matrices, node_feature_matrices, difficulties)
    # print(val_loss)
    # difficulty_categories = convert_to_categorical(difficulties, difficulties.min(), difficulties.max(), 20)
    # print(difficulty_categories.min(), difficulty_categories.max())