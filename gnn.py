import numpy as np 
import pandas as pd 
import torch

def data_load_in():
    df_train = pd.read_csv('data/csvs/train.csv')
    df_nodes = pd.read_csv('data/csvs/nodes.csv')
    print(df_train.head(), df_nodes)
    print(df_nodes.hold_type.unique())

def train():
    pass

if __name__ == "__main__":
    data_load_in()