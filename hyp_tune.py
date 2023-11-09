from models import *
import mlflow
from gnn import run_training_and_evaluation, data_loading
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


def run_training_and_evaluation(train_loader, test_loader, hidden_dim1=256, hidden_dim2=128, rnn_hidden_dim=256, num_epochs=100, lr=0.01, dropout_rate=0.5):
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("length train loader", len(train_loader))
    mlflow.log_param("loss", "MSE")
    input_dim = 100
    mlflow.log_param("hidden_dim1", hidden_dim1)
    mlflow.log_param("hidden_dim2", hidden_dim2)
    mlflow.log_param("rnn_hidden_dim", rnn_hidden_dim)
    model = SequentialRNNGNN(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, rnn_hidden_dim=rnn_hidden_dim, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr)
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
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
    plt.figure(figsize=(10, 5))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    val_loss_plot_filename = "val_loss_vs_pred.png"
    plt.savefig(val_loss_plot_filename)
    mlflow.log_artifact(val_loss_plot_filename)

    return best_val_loss

def main():
    # Define hyperparameter grid
    lrs = [0.001, 0.002, 0.005]
    dropout_rates = [0.3, 0.45, 0.5, 0.55]
    batch_sizes = [16, 32, 64]
    hidden_dims = [(256, 128), (512, 256), (128, 64)]
    num_epochs = 200
    rnn_hidden_dims = [128, 256]
    train_data_list, test_data_list = data_loading()
    mlflow.set_experiment("Hyperparameter_Tuning")

    for batch_size in batch_sizes:
        train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

        for hidden_dim1, hidden_dim2 in hidden_dims:
            for rnn_hidden_dim in rnn_hidden_dims:
                for dropout in dropout_rates:
                    for lr in lrs:
                        with mlflow.start_run():
                            mlflow.log_params({
                                'learning_rate': lr,
                                'dropout_rate': dropout,
                                'batch_size': batch_size,
                                'hidden_dim1': hidden_dim1,
                                'hidden_dim2': hidden_dim2,
                                'rnn_hidden_dim': rnn_hidden_dim
                            })
                            best_val_loss = run_training_and_evaluation(
                                train_loader, test_loader,
                                hidden_dim1, hidden_dim2, rnn_hidden_dim,
                                num_epochs, lr, dropout
                            )
                            mlflow.log_metric("best_val_loss", best_val_loss * 3.7927530348225766)
                            mlflow.end_run()

                        
if __name__ == "__main__":
    main()