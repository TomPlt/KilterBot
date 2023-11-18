from tqdm import tqdm 
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import re
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.sparse import save_npz
from scipy.spatial.distance import cdist
import ast
import sqlite3
import torch
from scipy.sparse import csr_matrix
from gnn import adjacency_to_edge_index


# Constants
kilter_xlim = 144
kilter_ylim = 156

# Function to extract hold data from a frame string
def extract_hold_data(frame_str):
    color_strings = re.findall(r'\d+', frame_str)
    return [int(color_str) for color_str in color_strings]

def get_coordinates(id_list, df_holes, normalize=False):
    coordinates = []
    for id_val in id_list:
        coords = df_holes[df_holes['id'] == id_val][['x', 'y']].values.tolist()
        if normalize:
            if coords:
                coords[0][0] /= kilter_xlim
                coords[0][1] /= kilter_ylim
        coordinates.extend(coords)
    return coordinates

# Load JSON data
def load_json_data():
    with open("data/jsons/specific_landmarks_sequence.json", "r") as file:
        vid_lm = json.load(file)
    with open("data/jsons/holds.json", "r") as file:
        vid_lm_holds = json.load(file)
    return vid_lm, vid_lm_holds

# Preprocess climbs data
def preprocess_climbs_data(df_climbs, df_holes):
    df_climbs['ids'] = df_climbs['frames'].apply(lambda x: extract_hold_data(x)[0::2])
    df_climbs['normalized_coordinates'] = df_climbs['ids'].apply(lambda x: get_coordinates(x, df_holes, normalize=True))
    df_climbs['colors'] = df_climbs['frames'].apply(lambda x: extract_hold_data(x)[1::2])
    return df_climbs

# Create a color mapping dictionary
def create_color_mapping(df_colors):
    color_mapping = df_colors.set_index('id')['full_name'].to_dict()
    return color_mapping

# Map hold types to climbs DataFrame
def map_hold_types(df_climbs, color_mapping):
    df_climbs['hold_type'] = df_climbs.colors.apply(
        lambda x: [
            color_mapping.get(color)
            if color else None
            for color in x
        ]
    )
    return df_climbs

# Organize holds into a dictionary
def organize_holds(df_climbs):
    kilter_holds = {}
    for counter, i in enumerate(df_climbs.hold_type[1]):
        if i in kilter_holds.keys():
            kilter_holds[i].append(df_climbs.normalized_coordinates[1][counter])
        else:
            kilter_holds[i] = [df_climbs.normalized_coordinates[1][counter]]
    return kilter_holds

# Perform clustering and plot
def perform_clustering_and_plot(hold_dict, n_clusters_dict, kilter_holds, eps=0.05, min_samples=2):
    colors = {'Start': '#00DD00', 'Middle': '#00FFFF', 'Finish': '#FF00FF'}
    cluster_centers_dict = {}
    
    plt.figure(figsize=(10, 6))
    
    for hold_type, coords in hold_dict.items():
        coords_array = np.array(coords)
        coords_array[:, 1] = 1 - coords_array[:, 1]  # Flip Y-axis
        n_clusters = n_clusters_dict.get(hold_type, 1)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coords_array)
        non_outliers = labels != -1
        
        # Use only non_outliers if they exist, otherwise use all points
        coords_to_use = non_outliers if np.sum(non_outliers) > 0 else None
        kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000, random_state=42)
        clusters = kmeans.fit_predict(coords_array[coords_to_use]) if coords_to_use is not None else kmeans.fit_predict(coords_array)
        cluster_centers = kmeans.cluster_centers_
        
        # Update cluster_centers_dict and plot
        cluster_centers_dict[hold_type] = cluster_centers
        plt.scatter(coords_array[:, 0], coords_array[:, 1], c=colors.get(hold_type, 'k'), label=hold_type, alpha=1, edgecolors='w')
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=colors.get(hold_type, 'k'), s=300, alpha=1, marker='o')

    # Matching Predicted and Real Hold Positions
    assignments = {}
    cluster_centers = {key: val[:, :2] for key, val in cluster_centers_dict.items()}
    
    for category, points1 in cluster_centers.items():
        points2 = np.array(kilter_holds[category])
        
        # Compute distance matrix and find optimal assignment
        cost_matrix = cdist(points1, points2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments[category] = (row_ind, col_ind)
        
        # Plotting
        plt.scatter(points2[:, 0], points2[:, 1], c=colors.get(category, 'k'), s=100, label=f'{category} Real', marker='o', edgecolors='w')
        
        for point1, point2 in zip(row_ind, col_ind):
            dx = points2[point2, 0] - points1[point1, 0]
            dy = points2[point2, 1] - points1[point1, 1]
            plt.arrow(points1[point1, 0], points1[point1, 1], dx, dy, 
                    shape='full', color='k', length_includes_head=True, 
                    head_width=0.02, linewidth=0.3, alpha=0.7)

    plt.title('Clustering and Matching of Hold Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
    return cluster_centers_dict, assignments

# Function to create a graph from clustering results
def create_graph_from_clustering(assignments, kilter_holds, right_hand_sequence, left_hand_sequence):
    df_train = pd.read_csv('data/csvs/train.csv')
    df_nodes = pd.read_csv('data/csvs/nodes.csv')
    row = df_train.loc[1]
    coordinates = ast.literal_eval(row['coordinates'])
    nodes = ast.literal_eval(row['nodes'])
    hold_variants = ast.literal_eval(row['hold_type'])
    real_pos = {}
    for hold_type, (row_ind, col_ind) in assignments.items():
        real_points = np.array(kilter_holds[hold_type])
        for i, j in zip(row_ind, col_ind):
            real_pos[f"{hold_type}_{i}"] = [int(i) for i in list(np.multiply(real_points[j, :2], [kilter_xlim, kilter_ylim]))]
    coord_dict = {node_id: coord for node_id, coord in zip(nodes, coordinates)}
    joined_dict = {}
    for key1, value1 in real_pos.items():
        for key2, value2 in coord_dict.items():
            if value1 == value2:
                joined_dict[key1] = key2
    G = nx.DiGraph()
    row = df_train.loc[1]
    coordinates = ast.literal_eval(row['coordinates'])
    nodes = ast.literal_eval(row['nodes'])
    hold_variants = ast.literal_eval(row['hold_type'])
    # Adding nodes
    for i, node_id in enumerate(nodes):
        node_features = df_nodes.loc[node_id].to_dict()
        G.add_node(node_id, coordinates=coord_dict[node_id], hold_variant=hold_variants[i], **node_features)

    # Adding edges
    for sequence, hand in [(right_hand_sequence, "Right"), (left_hand_sequence, "Left")]:
        for i in range(len(sequence) - 1):
            start_node = joined_dict.get(sequence[i])
            end_node = joined_dict.get(sequence[i+1])
            if start_node is not None and end_node is not None:
                G.add_edge(start_node, end_node, hand=hand)
    return G, real_pos

def visualize_graph(graphs: list, index):
    if index:
        G = graphs[index]
    else:
        G = graphs[0]
    pos = nx.get_node_attributes(G, 'coordinates')
    
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=15, font_weight='bold')
    
    for node, attrs in G.nodes(data=True):
        s = f"hold_variant: {attrs['hold_variant']}\n" + \
            "\n".join([f"{key}: {value}" for key, value in attrs.items() if key not in ['coordinates', 'hold_variant']])
        plt.annotate(s, (pos[node][0],pos[node][1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    
    plt.title('Graph Visualization with Features')
    plt.show()

def mirror_angle(angle):
    if 0 <= angle < 180:
        return 180 - angle
    else:
        return 540 - angle

def create_mirrored_graph(G):
    G_mirrored = nx.DiGraph()

    # Adding mirrored nodes
    for node, data in G.nodes(data=True):
        x_old, y = data['coordinates']
        x_new = kilter_xlim/2 + (kilter_xlim/2 - x_old)
        mirrored_coordinates = (x_new, y)
        mirrored_screw_angle = mirror_angle(data.get('screw_angle', 0))

        mirrored_node_features = data.copy()
        mirrored_node_features['coordinates'] = mirrored_coordinates
        mirrored_node_features['screw_angle'] = mirrored_screw_angle

        G_mirrored.add_node(node, **mirrored_node_features)

    # Adding edges (since they are the same, just mirrored)
    for u, v, data in G.edges(data=True):
        G_mirrored.add_edge(u, v, **data)

    return G_mirrored

def get_hand_sequence(cluster_centers_dict, vid_lm):
    # Step 1: Initialize Sequences and Nearest Holds
    right_hand_sequence = []
    left_hand_sequence = []
    nearest_holds_left = []
    nearest_holds_right = []


    # Flags to indicate if start and middle holds have been grabbed
    start_grabbed = False
    middle_grabbed = False

    # Holds
    start_holds = cluster_centers_dict["Start"][:,:2]
    finish_holds = cluster_centers_dict["Finish"][:,:2]
    middle_holds = cluster_centers_dict["Middle"][:,:2]

    # Step 2: Check for Start Holds
    for counter, i in enumerate(vid_lm):
        right_hand = np.array([i[1]['x'], abs(1-i[1]['y'])])
        left_hand = np.array([i[0]['x'], abs(1-i[0]['y'])])

        # Calculate distance to start holds
        dist_start_right = np.linalg.norm(start_holds - right_hand, axis=1)
        dist_start_left = np.linalg.norm(start_holds - left_hand, axis=1)

        # If not all start holds are grabbed
        if not start_grabbed:
            for idx, distance in enumerate(dist_start_right):
                if distance <= 0.05:
                    start_id = f"Start_{idx}"
                    if start_id not in right_hand_sequence:
                        right_hand_sequence.append(start_id)
                        
            for idx, distance in enumerate(dist_start_left):
                if distance <= 0.05:
                    start_id = f"Start_{idx}"
                    if start_id not in left_hand_sequence:
                        left_hand_sequence.append(start_id)
                        
            if len(right_hand_sequence) + len(left_hand_sequence) >= len(start_holds):
                start_grabbed = True


        dist_middle = np.linalg.norm(middle_holds - left_hand, axis=1)
        nearest_middle_index = np.argmin(dist_middle)
        dist_middle_right = np.linalg.norm(middle_holds - right_hand, axis=1)
        nearest_middle_index_right = np.argmin(dist_middle_right)

        if dist_middle[nearest_middle_index] <= 0.05 and nearest_middle_index not in nearest_holds_left and "Finish" not in "".join(left_hand_sequence):
            nearest_holds_left.append(nearest_middle_index)
            left_hand_sequence.append(f"Middle_{nearest_middle_index}")
        if dist_middle_right[nearest_middle_index_right] <= 0.05 and nearest_middle_index_right not in nearest_holds_right and "Finish" not in "".join(right_hand_sequence):
            nearest_holds_right.append(nearest_middle_index_right)
            right_hand_sequence.append(f"Middle_{nearest_middle_index_right}")

        dist_finish_right = np.linalg.norm(finish_holds - right_hand, axis=1)
        dist_finish_left = np.linalg.norm(finish_holds - left_hand, axis=1)
        
        for idx, distance in enumerate(dist_finish_right):
            if distance <= 0.07:
                finish_id_right = f"Finish_{idx}"
                if finish_id_right not in right_hand_sequence:
                    right_hand_sequence.append(finish_id_right)
                    
        for idx, distance in enumerate(dist_finish_left):
            if distance <= 0.07:
                finish_id_left = f"Finish_{idx}"
                if finish_id_left not in left_hand_sequence:
                    left_hand_sequence.append(finish_id_left)
        if "Finish" in right_hand_sequence and "Finish" in left_hand_sequence:
            break

    return left_hand_sequence, right_hand_sequence

# Main function to process data
def process_data():
    df_climbs_angles = pd.read_csv('data/csvs/climbs_with_angles.csv', low_memory=False)
    df_climbs = pd.read_csv('data/csvs/climbs.csv')
    df_holes = pd.read_csv('data/csvs/holes.csv')
    df_colors = pd.read_csv('data/csvs/placement_roles.csv')
    df_vscale = pd.read_csv('data/csvs/vscale.csv')

    df_climbs = preprocess_climbs_data(df_climbs, df_holes)
    vid_lm, vid_lm_holds = load_json_data()
    color_mapping = create_color_mapping(df_colors)
    df_climbs = map_hold_types(df_climbs, color_mapping)
    kilter_holds = organize_holds(df_climbs)
    n_clusters_dict = {'Start': len(kilter_holds['Start']), 'Middle': len(kilter_holds['Middle']), 'Finish': len(kilter_holds['Finish'])}
    cluster_centers_dict, assignments = perform_clustering_and_plot(vid_lm_holds, n_clusters_dict, kilter_holds)
    left_hand_sequence, right_hand_sequence = get_hand_sequence(cluster_centers_dict, vid_lm)
    G, real_pos = create_graph_from_clustering(assignments, kilter_holds, right_hand_sequence, left_hand_sequence)
    visualize_graph([G], 0)
    G_mirrored = create_mirrored_graph(G)
    visualize_graph([G_mirrored], 0)

# Function to get edge data from SQLite database
def get_edge_data_from_db(graph_index):
    # Connect to SQLite database
    conn = sqlite3.connect('../edgeapp/edges.db')
    # Execute query to fetch edge data for the specific graph index
    query = f"SELECT * FROM edges WHERE graph_index = {graph_index}"
    df_edges = pd.read_sql(query, conn)
    conn.close()
    return df_edges


def verify_features(nodes, df_nodes):
    feature_keys = set(df_nodes.columns)  # Assuming df_nodes has consistent columns
    for node in nodes:
        node_features = set(df_nodes.loc[node].to_dict().keys())
        if feature_keys != node_features:
            missing = feature_keys - node_features
            extra = node_features - feature_keys
            return node, missing, extra
    return None, None, None

def build_and_save_graphs_with_features(index: int):
    df_train = pd.read_csv('data/csvs/train.csv')
    df_nodes = pd.read_csv('data/csvs/nodes.csv')
    # Normalize the screw_angle and x, y coordinates
    df_nodes['norm_screw_angle'] = df_nodes['screw_angle'] / 360
    df_nodes['norm_x'] = df_nodes['x'] / df_nodes['x'].max()
    df_nodes['norm_y'] = df_nodes['y'] / df_nodes['y'].max()
    df_nodes = pd.get_dummies(df_nodes, columns=['sku', 'hold_type'])
    df_nodes.drop(columns=['name', 'screw_angle', 'x', 'y'], inplace=True)
    # df_train = pd.get_dummies(df_train, columns=['hold_type'])

    # Ensure all hold_type_columns exist in the DataFrame
    hold_type_columns = [
        'hold_type_Start',
        'hold_type_Middle', 
        'hold_type_Finish', 
        'hold_type_Foot Only', 
    ]
    for col in hold_type_columns:
        if col not in df_train.columns:
            df_train[col] = 0

   
    # Create interaction terms for norm_screw_angle and SKU dummies
    for sku_col in [col for col in df_nodes.columns if col.startswith('sku_')]:
        df_nodes[f'{sku_col}_angle_interaction'] = df_nodes[sku_col] * df_nodes['norm_screw_angle']
    
    row = df_train.loc[index]
    climbs = pd.read_csv('data/csvs/climbs.csv')
    # get the name and difficulty of the climb matching the uuid
    climb_name = climbs[climbs['uuid'] == row['uuid']]['name'].values[0]
    climb_difficulty = row['difficulty']
    coordinates = ast.literal_eval(row['coordinates'])
    nodes = ast.literal_eval(row['nodes'])
    hold_variants = ast.literal_eval(row['hold_type'])
    if None in hold_variants:
        return
    map_hold_vars = {'Start': 0, 'Middle': 1, 'Finish': 2, 'Foot Only': 3}
    if not any(map_hold_vars[i] in [0, 2] for i in hold_variants):
        return

    hold_vars_list = [map_hold_vars[i] for i in hold_variants]
    coord_dict = {node: coord for node, coord in zip(nodes, coordinates)}
    G = nx.DiGraph()
    assert len(nodes) == len(hold_vars_list)
    assert len(nodes) == len(coordinates)
    nodes_dict = {}
    for i, node in enumerate(nodes):
        node_features = df_nodes.loc[node]
        G.add_node(node, coordinates=coord_dict[node], hold_variant=hold_vars_list[i], **node_features)
        nodes_dict[node] = i
    # Assuming get_edge_data_from_db(index) is a function you've defined to get the edge data
    df_edges = get_edge_data_from_db(index)
    df_edges = df_edges.sort_values(by='edge_index')
    for _, edge_row in df_edges.iterrows():
        foothold_counter = 0    
        features_temp= [i[1]['coordinates'] for i in G.nodes(data=True)]
        for i in G.nodes(data=True):
            if i[0] == edge_row['start_node']:
                start_node_coord = i[1]['coordinates']
                break
        for i in features_temp: 
            if i[1] < start_node_coord[1] and np.linalg.norm(np.array(i) - np.array(start_node_coord)) <= 80 :
                foothold_counter += 1
        G.add_edge(edge_row['start_node'], edge_row['end_node'], footholds=foothold_counter)
    adjacency_matrix = nx.adjacency_matrix(G)
    save_npz(f"data/npzs/adjacency_mtrx/{index}.npz", adjacency_matrix)
    edge_index = adjacency_to_edge_index(adjacency_matrix.todense())
    features_list = []
    for node in G.nodes(data=True):
        features_list.append(list(node[1].values())[1:])
        if len(features_list[-1]) != len(features_list[0]):
            raise ValueError(f"Node {node} has a different number of features.")
    try:
        node_feature_array = np.array(features_list)
    except ValueError as e:
        print(f"Error constructing feature array: {e}")
        raise
    # print(df_edges)
    node_feature_matrix = csr_matrix(node_feature_array)
    save_npz(f"data/npzs/node_feature_mtrx/{index}.npz", node_feature_matrix)
    
    ordered_edges = df_edges[['start_node', 'end_node']].values.tolist()
    indexed_edges = []
    foothold_attributes = []
    for start_node, end_node in ordered_edges:
    # Retrieve the foothold attribute from the edge
        foothold = G[start_node][end_node]['footholds'] if G.has_edge(start_node, end_node) else 0
        foothold_attributes.append(foothold)
    foothold_attribute_tensor = torch.tensor(foothold_attributes, dtype=torch.long)
    torch.save(foothold_attribute_tensor, f"data/tensors/foothold_attributes_{index}.pt")

    for start, end in ordered_edges:
        edge_tuple = (nodes_dict[start], nodes_dict[end])
        if edge_tuple not in indexed_edges:
            indexed_edges.append(edge_tuple)
    edge_index_tensor_ordered = torch.tensor(indexed_edges, dtype=torch.long)
    torch.save(edge_index_tensor_ordered, f"data/tensors/edge_sequence_{index}.pt")
    # save also the difficulty and nameto a json 
    climb_info = {'name': climb_name, 'difficulty': climb_difficulty}
    with open(f"data/jsons/climb_info/{index}.json", "w") as file:
        json.dump(climb_info, file)
    return G

if __name__ == "__main__":
    index = 4210
    for i in tqdm(range(0, index+1)):
        build_and_save_graphs_with_features(i)
        # exit()
