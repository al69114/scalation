import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Graphs are saved next to this script, regardless of where it is run from
GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "graphs")

# Make results more repeatable
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 5L means input + 3 hidden layers + output
class NeuralNet5L(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation="relu"):
        super().__init__()

        activation_layer = nn.ReLU if activation == "relu" else nn.Tanh

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            activation_layer(),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation_layer(),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            activation_layer(),

            nn.Linear(hidden_sizes[2], 1)
        )

    def forward(self, x):
        return self.network(x)


def load_dataset(path, target):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(df.columns.tolist())
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in {path}")

    # Convert columns to numbers and remove rows with missing/non-numeric values
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    y = df[target].to_numpy(dtype=np.float32)
    x = df.drop(columns=[target]).to_numpy(dtype=np.float32)

    return x, y


def train_model(x_train, y_train, hidden_sizes, activation, learning_rate,
                batch_size, epochs=300, seed=42, x_val=None, y_val=None, patience=20,
                history=None):
    # When validation data is given, track the validation MSE each epoch and stop
    # early once it has not improved for `patience` epochs.
    # If a history dict is given, the training and validation MSE of every epoch are saved in it
    set_seed(seed)

    model = NeuralNet5L(
        input_size=x_train.shape[1],
        hidden_sizes=hidden_sizes,
        activation=activation
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_mse = float("inf")
    best_epoch = epochs
    wait = 0

    for epoch in range(epochs):
        model.train()

        for batch_x, batch_y in loader:
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if x_val is not None:
            val_mse = np.mean((y_val - predict(model, x_val)) ** 2)

            if history is not None:
                history["train"].append(np.mean((y_train - predict(model, x_train)) ** 2))
                history["val"].append(val_mse)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch + 1
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                break

    return model, best_epoch, best_val_mse


def predict(model, x):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(x_tensor).squeeze().numpy()

    return predictions


def calculate_qof(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    total_sum_squares = np.sum((actual - np.mean(actual)) ** 2)
    residual_sum_squares = np.sum((actual - predicted) ** 2)

    r2 = (
        1 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )

    return {"r2": r2, "mse": mse, "rmse": rmse}


def tune_hyperparameters(x, y, seed=42):
    # This is the inner split used to choose hyperparameters
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.20, random_state=seed
    )

    # Fit scaling using only the inner training data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    # Standardize the target too; hidden layers train poorly on raw targets (~125 dB for Airfoil)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_var = y_scaler.scale_[0] ** 2  # converts scaled MSE back to original units

    # Candidate settings to compare
    hidden_sizes_options = [
        (8, 8, 8),
        (16, 16, 16),
        (32, 16, 8),
        (64, 32, 16),
    ]
    activation_options = ["relu", "tanh"]
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    batch_sizes = [16, 32, 64]

    best_params = None
    best_history = None
    best_mse = float("inf")

    for hidden_sizes in hidden_sizes_options:
        for activation in activation_options:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    history = {"train": [], "val": []}
                    _, best_epoch, val_mse = train_model(
                        x_train,
                        y_train,
                        hidden_sizes,
                        activation,
                        learning_rate,
                        batch_size,
                        seed=seed,
                        x_val=x_val,
                        y_val=y_val,
                        history=history
                    )
                    val_mse *= y_var

                    print(
                        f"Trying layers={hidden_sizes}, activation={activation}, "
                        f"learning_rate={learning_rate}, batch_size={batch_size}: "
                        f"validation MSE={val_mse:.5f}, best epoch={best_epoch}"
                    )

                    if np.isfinite(val_mse) and val_mse < best_mse:
                        best_mse = val_mse
                        best_params = {
                            "hidden_sizes": hidden_sizes,
                            "activation": activation,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "validation_mse": val_mse,
                            "best_epoch": best_epoch,
                            "epochs": len(history["val"])
                        }
                        # loss curves in original units for the loss plot
                        best_history = {k: [v * y_var for v in vals] for k, vals in history.items()}

    return best_params, best_history

def plot_actual_vs_predicted(name, y_train, train_predictions, y_val, val_predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, actual, predicted, split in [
        (axes[0], y_train, train_predictions, "Training"),
        (axes[1], y_val, val_predictions, "Validation")
    ]:
        ax.scatter(actual, predicted, alpha=0.6)

        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], "r--")

        ax.set_title(f"{name}: {split}")
        ax.set_xlabel("Actual values")
        ax.set_ylabel("Predicted values")

    fig.tight_layout()
    os.makedirs(GRAPH_DIR, exist_ok=True)
    filename = os.path.join(GRAPH_DIR, name.lower().replace(" ", "_") + "_5l_graph.png")
    fig.savefig(filename, dpi=300)
    plt.show()

def plot_loss(name, history):
    plt.figure(figsize=(6, 3))
    plt.plot(history["train"], label="Training Loss")
    plt.plot(history["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{name} - 5L Neural Network Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, name.lower().replace(" ", "_") + "_5l_loss.png"), dpi=150)
    plt.show()

def plot_residuals(name, y_val, val_predictions):
    residuals = y_val - val_predictions
    plt.figure(figsize=(7, 5))
    plt.scatter(val_predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"{name} - 5L Validation Residual Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, name.lower().replace(" ", "_") + "_5l_residual.png"), dpi=150)
    plt.show()

def run_experiment(name, path, target, seed=42):
    print(f"\n========== {name}: 5L PYTORCH NEURAL NETWORK ==========")

    x, y = load_dataset(path, target)
    print(f"Observations: {len(y)}")
    print(f"Predictors: {x.shape[1]}")

    # Outer split: keep this validation data separate until final evaluation
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.20, random_state=seed
    )

    # Same inner split seed as the 2L and 4L models
    best, history = tune_hyperparameters(x_train, y_train, seed)
    print("\nSelected hyperparameters:")
    print(best)

    # Fit the final scalers only on the full outer training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Train the final model using the selected settings for the best number of epochs
    model, _, _ = train_model(
        x_train_scaled,
        y_train_scaled,
        best["hidden_sizes"],
        best["activation"],
        best["learning_rate"],
        best["batch_size"],
        epochs=best["best_epoch"],
        seed=seed
    )

    # Convert predictions back to the original scale for QoF
    train_predictions = y_scaler.inverse_transform(predict(model, x_train_scaled).reshape(-1, 1)).ravel()
    val_predictions = y_scaler.inverse_transform(predict(model, x_val_scaled).reshape(-1, 1)).ravel()

    train_results = calculate_qof(y_train, train_predictions)
    val_results = calculate_qof(y_val, val_predictions)

    print("\nTraining results:")
    print(f"R²:   {train_results['r2']:.5f}")
    print(f"MSE:  {train_results['mse']:.5f}")
    print(f"RMSE: {train_results['rmse']:.5f}")

    print("\nFinal validation results:")
    print(f"R²:   {val_results['r2']:.5f}")
    print(f"MSE:  {val_results['mse']:.5f}")
    print(f"RMSE: {val_results['rmse']:.5f}")
    plot_actual_vs_predicted(
    name, y_train, train_predictions, y_val, val_predictions
)
    plot_loss(name, history)
    plot_residuals(name, y_val, val_predictions)

    # Retrain with other seeds to check how much the validation R² depends on the random initialization
    seed_r2 = []
    for s in range(5):
        seed_model, _, _ = train_model(
            x_train_scaled, y_train_scaled, best["hidden_sizes"], best["activation"],
            best["learning_rate"], best["batch_size"], epochs=best["best_epoch"], seed=s
        )
        seed_pred = y_scaler.inverse_transform(predict(seed_model, x_val_scaled).reshape(-1, 1)).ravel()
        seed_r2.append(calculate_qof(y_val, seed_pred)["r2"])
    print(f"\nValidation R² over seeds 0-4: mean={np.mean(seed_r2):.4f}, sd={np.std(seed_r2):.4f}, "
          f"min={min(seed_r2):.4f}, max={max(seed_r2):.4f}")

    return {
        "Dataset": name,
        "Hidden Layers": str(best["hidden_sizes"]),
        "Activation": best["activation"],
        "Learning Rate": best["learning_rate"],
        "Batch Size": best["batch_size"],
        "Inner-Validation MSE": best["validation_mse"],
        "Epochs": best["epochs"],
        "Best Epoch": best["best_epoch"],
        "Train R2": train_results["r2"],
        "Validation R2": val_results["r2"],
        "Train MSE": train_results["mse"],
        "Validation MSE": val_results["mse"],
        "Train RMSE": train_results["rmse"],
        "Validation RMSE": val_results["rmse"],
        "Seed Mean Validation R2": np.mean(seed_r2),
        "Seed SD Validation R2": np.std(seed_r2),
    }



if __name__ == "__main__":
    datasets = [
        ("Airfoil", "data/airfoil.csv", "scaled_sound_pressure_db"),
        ("Auto MPG", "data/auto_mpg.csv", "mpg"),
        ("Concrete", "data/concrete.csv", "compressive_strength_mpa"),
    ]

    results = [run_experiment(name, path, target) for name, path, target in datasets]

    # Save all numbers used in the report
    results_path = os.path.join(os.path.dirname(GRAPH_DIR), "5L_results.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")