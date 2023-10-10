import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re

kilter_xlim = 140
kilter_ylim = 152

def extract_hold_data(frame_str):
    color_strings = re.findall(r'\d+', frame_str)
    return [int(color_str) for color_str in color_strings]

def get_coordinates(id_list, df_holes, normalize=False):
    coordinates = []
    for id_val in id_list:
        coords = df_holes[df_holes['id'] == id_val][['x', 'y']].values.tolist()
        if normalize: 
            if coords != []:
                coords[0][0] /= kilter_xlim
                coords[0][1] /= kilter_ylim
        coordinates.extend(coords)
    return coordinates

def color_mapping(index, df_colors):
    color = df_colors.loc[df_colors.id == index].screen_color.values
    if color: return f"#{color[0]}"
    else: return f"error index"

def create_graph(climb_data):
    G = nx.Graph()
    node_idx = 0  # to keep a unique identifier for each node
    coordinates = []  # to store coordinates for distance computation

    for hold_type, coords_list in climb_data.items():
        for coord in coords_list:
            G.add_node(node_idx, coord=coord, type=hold_type)
            coordinates.append(coord)  # add to coordinates list
            node_idx += 1  # increment for the next node

    distances = np.linalg.norm(np.array(coordinates)[:, None, :] - np.array(coordinates)[None, :, :], axis=-1)
    for idx, node_dists in enumerate(distances):
        nearest_neighbors = np.argsort(node_dists)[1:3]
        for neighbor_idx in nearest_neighbors:
            G.add_edge(idx, neighbor_idx)
    
    return G

def compute_hold_id(x, y):
    # Check large holds
    if x % 8 == 0 and y % 8 == 0:
        nx = x // 8
        ny = y // 8
        return f"L_{nx}_{ny}"  # Large hold

    # Check small holds
    elif y == 4 and (x - 4) % 8 == 0:
        n = (x - 4) // 8
        return f"S1_{n}"  # Small hold type 1
    
    elif (y - 20) % 16 == 0 and (x - 4) % 16 == 0 and y != 4 and y <= 132: 
        ny = (y - 20) // 16
        nx = (x - 4) // 16
        return f"S2_{ny}_{nx}"  # Small hold type 2
    
    elif (y - 28) % 16 == 0 and (x - 12) % 16 == 0 and y != 12 and y <= 124:
        ny = (y - 28) // 16
        nx = (x - 12) // 16
        return f"S3_{ny}_{nx}"  # Small hold type 3
    
    else:
        return None  # Invalid coordinates

def generate_id_mapping(max_x=kilter_xlim, max_y=kilter_ylim):
    next_id = 0
    id_mapping = {}
    for x in range(4, max_x + 1):
        for y in range(4, max_y + 1):
            str_id = compute_hold_id(x, y)
            if str_id is not None and str_id not in id_mapping:
                id_mapping[str_id] = next_id
                next_id += 1
    return id_mapping

def get_numeric_id(x, y, id_mapping):
    str_id = compute_hold_id(x, y)
    return id_mapping.get(str_id, None)

def main():
    id_mapping = generate_id_mapping()
    print(id_mapping)
    # for i in test:
    #     print(compute_hold_id(i[0], i[1]))  
    # print(compute_hold_id(0, 152))

#     df_climbs = pd.read_csv('data/csvs/climbs.csv')
#     df_colors = pd.read_csv('data/csvs/placement_roles.csv')
#     df_holes = pd.read_csv('data/csvs/holes.csv')
#     color_mapping = df_colors.set_index('id')['full_name'].to_dict()

#     df_climbs['ids'] = df_climbs['frames'].apply(lambda x: extract_hold_data(x)[0::2])
#     df_climbs['colors'] = df_climbs['frames'].apply(lambda x: extract_hold_data(x)[1::2])
#     df_climbs['coordinates'] = df_climbs['ids'].apply(lambda x: get_coordinates(x, df_holes))
#     df_climbs['hold_type'] = df_climbs.colors.apply(
#         lambda x: [
#             color_mapping.get(color) 
#             if color else None
#             for color in x
#         ]
#     )
#     print(df_climbs.coordinates[1])

    # for index, row in df_climbs.iterrows():
    #     climb_data = {}
    #     for counter, i in enumerate(row['hold_type']):
    #         if i in climb_data.keys():
    #             climb_data[i].append(row['coordinates'][counter])
    #         else:
    #             climb_data[i] = [row['coordinates'][counter]]
        
    #     # Creating a graph and obtaining the adjacency matrix
    #     G = create_graph(climb_data)
    #     adjacency_matrix = nx.to_numpy_array(G)
    #     print(adjacency_matrix)

if __name__ == "__main__":
    main()