import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re

kilter_xlim = 140
kilter_ylim = 152
# 12'x12' Original Layout
# 323 unique bolt-on holds, 153 screw-on holds, 1060 LEDs/kit

def map_hold_skus(skus):
    sku_map = {}
    for _, row in skus.iterrows():
        num_range = create_range(row['NUMBERS'])
        for num in num_range:
            sku_map[str(num)] = row['SKU']
    return sku_map


def map_hold_angles(screw_angles):
    if x.startswith('L'):
        x, y = x[2:].split('_')
        return screw_angles.iloc[int(y)-1, int(x)-1]
    else:
        return float(0)

def create_range(s):
    if s[0].isdigit():
        start, end = map(int, s.split('-'))
        return list(range(start, end+1))
    else:
        start, end = s.split('-')
        start_num = int(start[1:])
        end_num = int(end[1:])
        char = start[0]
        return [f"{char}{i}" for i in range(start_num, end_num+1)]

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
        return f"S2_{nx}_{ny}"  # Small hold type 2
    
    elif (y - 28) % 16 == 0 and (x - 12) % 16 == 0 and y != 12 and y <= 124:
        ny = (y - 28) // 16
        nx = (x - 12) // 16
        return f"S3_{nx}_{ny}"  # Small hold type 3
    
    else:
        return None  # Invalid coordinates

def generate_id_mapping(max_x=kilter_xlim, max_y=kilter_ylim):
    next_id = 0
    coord_list = []
    id_mapping = {}
    for x in range(4, max_x + 1):
        for y in range(4, max_y + 1):
            str_id = compute_hold_id(x, y)
            if str_id is not None and str_id not in id_mapping:
                id_mapping[str_id] = next_id
                coord_list.append([x, y])
                next_id += 1
    return id_mapping, coord_list

def get_numeric_id(x, y, id_mapping):
    str_id = compute_hold_id(x, y)
    return id_mapping.get(str_id, None)

