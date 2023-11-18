from models import *
import mlflow
from gnn import run_training_and_evaluation, data_loading
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import numpy as np
import optuna
import logging
import torch

logging.basicConfig(level=logging.INFO)

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
def run_training_and_evaluation(train_loader, test_loader, hidden_dim1=256, hidden_dim2=128, rnn_hidden_dim=128, lr=0.01, dropout_rate=0.5, weight_decay=5e-4, num_epochs=100):
    # mlflow.log_param("num_epochs", num_epochs)
    input_dim = 100
    # mlflow.log_param("hidden_dim1", hidden_dim1)
    # mlflow.log_param("hidden_dim2", hidden_dim2)
    # mlflow.log_param("rnn_hidden_dim", rnn_hidden_dim)
    model = SequentialRNNGNN(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, rnn_hidden_dim=rnn_hidden_dim, dropout_rate=dropout_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    # criterion is RMSE loss
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

        # Validation phase
        for batch in test_loader:
            with torch.no_grad():
                out = model(batch).squeeze(-1)
                loss = l1_loss_criterion(out, batch.y)
                total_val_loss += loss.item() * batch.num_graphs
                true_values.extend(batch.y.tolist())
                predicted_values.extend(out.tolist())

        avg_val_loss = total_val_loss / len(test_loader.dataset)
        val_loss_list.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        # mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        # mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    bins, mean_losses = calculate_mean_loss_per_bin(true_values, predicted_values)
   # Plotting True Values vs Predicted Values
    plt.figure(figsize=(10, 5))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig("true_vs_pred.png")
    # mlflow.log_artifact("true_vs_pred.png")

    # Plotting Mean Loss by Bins
    plt.figure(figsize=(10, 5))
    plt.bar(bins[:-1], mean_losses, width=np.diff(bins), align="edge", edgecolor='black')
    plt.title('Mean Loss by Bins of True Values')
    plt.xlabel('Bins of True Values')
    plt.ylabel('Mean Loss')
    plt.grid(True)
    plt.savefig("mean_loss_by_bins.png")
    # mlflow.log_artifact("mean_loss_by_bins.png")
    plt.close('all')
    return best_val_loss

def objective(trial):
    # Define hyperparameters using trial object
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    hidden_dim1 = trial.suggest_categorical("hidden_dim1", [256, 512, 128])
    hidden_dim2 = trial.suggest_categorical("hidden_dim2", [128, 256, 64])
    rnn_hidden_dim = trial.suggest_categorical("rnn_hidden_dim", [64, 128, 256])
    fold_data_lists = data_loading()
    fold_val_losses = []

    for train_data, test_data in fold_data_lists:
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Run the training and validation process for each fold
        best_val_loss = run_training_and_evaluation(
            train_loader, test_loader,
            hidden_dim1, hidden_dim2, rnn_hidden_dim,
            lr, dropout_rate, weight_decay, num_epochs=50)
        fold_val_losses.append(best_val_loss)
    
    avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)

    return avg_val_loss

def main():
    # Define hyperparameter grid
    lrs = [0.001, 0.002, 0.005]
    dropout_rates = [0.3, 0.45, 0.5, 0.55]
    batch_sizes = [32, 64]
    hidden_dims = [(256, 128), (512, 256), (128, 64)]
    num_epochs = 200  # You might want to reduce this for quicker iterations
    train_data_list, test_data_list = data_loading()
    mlflow.set_experiment("Hyperparameter Tuning 2")

    # Initialize counter for the number of runs
    total_runs = len(lrs) * len(dropout_rates) * len(batch_sizes) * len(hidden_dims)
    counter = 0

    for batch_size in batch_sizes:
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)
        
        for hidden_dim1, hidden_dim2 in hidden_dims:
            for dropout_rate in dropout_rates:
                for lr in lrs:
                    counter += 1
                    print(f"Running configuration {counter} of {total_runs}")
                    with mlflow.start_run():
                        mlflow.log_params({
                            'learning_rate': lr,
                            'dropout_rate': dropout_rate,
                            'batch_size': batch_size,
                            'hidden_dim1': hidden_dim1,
                            'hidden_dim2': hidden_dim2
                        })
                        best_val_loss = run_training_and_evaluation(
                            train_loader, test_loader,
                            hidden_dim1, hidden_dim2,
                            num_epochs, lr, dropout_rate
                        )
                        mlflow.log_metric("best_val_loss", best_val_loss)
                        mlflow.end_run()
                    print(f"Configuration {counter} of {total_runs} completed: Best Val Loss: {best_val_loss}")

if __name__ == "__main__":
    # Ensure that Optuna reuses the study if it exists, otherwise create a new one
    try:
        study = optuna.load_study(
            study_name="Hyperparameter Tuning RNN", 
            storage="sqlite:///hyptune_RNN.db"
        )
    except KeyError:
        # If the study does not exist, create a new one
        study = optuna.create_study(
            direction="minimize", 
            study_name="Hyperparameter Tuning RNN", 
            storage="sqlite:///hyptune_RNN.db",
            load_if_exists=True
        )

    study.optimize(objective, n_trials=100)