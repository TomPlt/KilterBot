from models import *
import mlflow
from gnn import run_training_and_evaluation, data_loading, calculate_mean_loss_per_bin
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import numpy as np
import optuna
import logging
import torch

logging.basicConfig(level=logging.INFO)

def objective(trial):
    # Define hyperparameters using trial object
    conv_layer_choices = {
        "GATConv": GATConv,
        "GCNConv": GCNConv,
        "ChebConv": lambda in_channels, out_channels, K=2: ChebConv(in_channels, out_channels, K=K),
        "TransformerConv": TransformerConv
    }
    k_default = 1  # Default value for K
    conv_type1 = trial.suggest_categorical("conv_type1", list(conv_layer_choices.keys()))
    conv_type2 = trial.suggest_categorical("conv_type2", list(conv_layer_choices.keys()))
    conv_type3 = trial.suggest_categorical("conv_type3", list(conv_layer_choices.keys()))

    conv_types = [conv_layer_choices[conv_type1], conv_layer_choices[conv_type2], conv_layer_choices[conv_type3]]
    k_values = [k_default if conv != "ChebConv" else 2 for conv in [conv_type1, conv_type2, conv_type3]]

    conv_types = [ChebConv, ChebConv, TransformerConv]
    k_values = [2, 2, 2]
    # conv_types = [conv_layer_choices[conv_type1], conv_layer_choices[conv_type2], conv_layer_choices[conv_type3]]
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    dropout_rate1 = trial.suggest_float("dropout_rate1", 0.1, 0.6)
    dropout_rate2 = trial.suggest_float("dropout_rate2", 0.1, 0.6)
    num_epochs = trial.suggest_int("num_epochs", 10, 200, step=10)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    hidden_dim1 = trial.suggest_categorical("hidden_dim1", [32, 64, 128, 256, 512])
    hidden_dim2 = trial.suggest_categorical("hidden_dim2", [32, 64, 128, 256, 512])
    hidden_dim3 = trial.suggest_categorical("hidden_dim3", [32, 64, 128, 256, 512])
    # rnn_hidden_dim = trial.suggest_categorical("rnn_hidden_dim", [64, 128, 256])
    fold_data_lists = data_loading()
    fold_val_losses = []

    for train_data, test_data in fold_data_lists:
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        # Run the training and validation process for each fold
        best_val_loss = run_training_and_evaluation(
            conv_types, k_values, train_loader, test_loader,
            hidden_dim1, hidden_dim2, hidden_dim3,
            lr, dropout_rate1, dropout_rate2, weight_decay, num_epochs)
        fold_val_losses.append(best_val_loss)
    return sum(fold_val_losses) / len(fold_val_losses)

if __name__ == "__main__":
    # Ensure that Optuna reuses the study if it exists, otherwise create a new one
    db = "sqlite:///hyptune_different_layers.db"
    name = "var_num_epochs"
    try:
        study = optuna.load_study(
            study_name=name, 
            storage=db,
        )
    except KeyError:
        # If the study does not exist, create a new one
        study = optuna.create_study(
            direction="minimize", 
            study_name=name, 
            storage=db,
            load_if_exists=True
        )

    study.optimize(objective)