import networkx as nx
from scipy.sparse import csr_matrix
import ast
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
from enum import Enum

class MtrxType(Enum):
    ADJACENCY_MATRIX = 0
    NODE_FEATURE_MATRIX = 1

def data_load_in():
    df_train = pd.read_csv('data/csvs/train.csv')
    df_nodes = pd.read_csv('data/csvs/nodes.csv')
    return df_train, df_nodes

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
    df_train, df_nodes = data_load_in()
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

if __name__ == "__main__":
    df_train, df_nodes = data_load_in()
    index = 0
    graph = load_mtrx(0, MtrxType.ADJACENCY_MATRIX)
    node_feature_mtrx = load_mtrx(0, MtrxType.NODE_FEATURE_MATRIX)
    print(graph.shape, node_feature_mtrx.shape)