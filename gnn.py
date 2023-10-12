import networkx as nx
import ast
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def data_load_in():
    df_train = pd.read_csv('data/csvs/train.csv')
    df_nodes = pd.read_csv('data/csvs/nodes.csv')
    return df_train, df_nodes

def generate_graphs():
    # If you use a list:
    graphs = []

    df_train, df_nodes = data_load_in()

    for index, row in df_train.iterrows():
        G = nx.Graph()  # Initialize a new graph for this row
        
        coordinates = ast.literal_eval(row['coordinates'])
        nodes = ast.literal_eval(row['nodes'])
        hold_variants = ast.literal_eval(row['hold_type'])
        
        for i, coord in enumerate(coordinates):
            node_id = nodes[i]  # Removed redundant ast.literal_eval
            hold_variant = hold_variants[i]
            
            # Add nodes and node attributes to the graph
            node_features = df_nodes.loc[node_id].to_dict()
            G.add_node(node_id, 
                    coordinates=coord, 
                    hold_variant=hold_variant, 
                    **node_features)

        # Calculate distances and create edges:
        for node_id in G.nodes(data=True):
            distances = {}
            for target_node_id in G.nodes(data=True):
                if node_id[0] != target_node_id[0]:
                    dist = np.linalg.norm(np.array(node_id[1]['coordinates']) - np.array(target_node_id[1]['coordinates']))
                    distances[target_node_id[0]] = dist
            # Get two nearest neighbors
            nearest_neighbors = sorted(distances.keys(), key=lambda x: distances[x])[:2]
            for neighbor in nearest_neighbors:
                G.add_edge(node_id[0], neighbor)
        graphs.append(G)
    return graphs

def visualize_graph(graphs: list, index):
    G = graphs[index]

    # Positioning nodes using 'coordinates'
    pos = nx.get_node_attributes(G, 'coordinates')

    # Drawing nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=15, font_weight='bold')

    # Annotating nodes with their features. Assuming features are 'hold_variant' and other node features
    for node, attrs in G.nodes(data=True):
        # Creating a string of features for demonstration. 
        # Adjust accordingly to your use case.
        s = f"hold_variant: {attrs['hold_variant']}\n" + \
            "\n".join([f"{key}: {value}" for key, value in attrs.items() if key not in ['coordinates', 'hold_variant']])
        
        plt.annotate(s, (pos[node][0],pos[node][1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

    plt.title('Graph Visualization with Features')
    plt.show()

generate_graphs()