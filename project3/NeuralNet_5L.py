import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
                batch_size, epochs=100, seed=42):
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

    for epoch in range(epochs):
        model.train()

        for batch_x, batch_y in loader:
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


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

    # Candidate settings to compare
    hidden_sizes_options = [
        (8, 8, 8),
        (16, 16, 16),
        (32, 16, 8),
        (64, 32, 16),
    ]
    activation_options = ["relu", "tanh"]
    learning_rates = [0.001, 0.01, 0.05]
    batch_sizes = [32, 64]

    best_params = None
    best_mse = float("inf")

    for hidden_sizes in hidden_sizes_options:
        for activation in activation_options:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    model = train_model(
                        x_train,
                        y_train,
                        hidden_sizes,
                        activation,
                        learning_rate,
                        batch_size,
                        seed=seed
                    )

                    val_predictions = predict(model, x_val)
                    val_mse = np.mean((y_val - val_predictions) ** 2)

                    print(
                        f"Trying layers={hidden_sizes}, activation={activation}, "
                        f"learning_rate={learning_rate}, batch_size={batch_size}: "
                        f"validation MSE={val_mse:.5f}"
                    )

                    if np.isfinite(val_mse) and val_mse < best_mse:
                        best_mse = val_mse
                        best_params = {
                            "hidden_sizes": hidden_sizes,
                            "activation": activation,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "validation_mse": val_mse
                        }

    return best_params

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
    filename = name.lower().replace(" ", "_") + "_5l_graph.png"
    fig.savefig(filename, dpi=300)
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

    best = tune_hyperparameters(x_train, y_train, seed + 1)
    print("\nSelected hyperparameters:")
    print(best)

    # Fit the final scaler only on the full outer training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    # Train the final model using the selected settings
    model = train_model(
        x_train_scaled,
        y_train,
        best["hidden_sizes"],
        best["activation"],
        best["learning_rate"],
        best["batch_size"],
        seed=seed
    )

    train_predictions = predict(model, x_train_scaled)
    val_predictions = predict(model, x_val_scaled)

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
    return best, train_results, val_results
    


if __name__ == "__main__":
    datasets = [
        ("Airfoil", "data/airfoil.csv", "scaled-sound-pressure"),
        ("Auto MPG", "data/auto_mpg.csv", "mpg"),
        ("Concrete", "data/concrete.csv", "Concrete compressive strength"),
    ]

    for name, path, target in datasets:
        run_experiment(name, path, target)