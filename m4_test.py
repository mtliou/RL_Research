import os
import torch
from collections import defaultdict
from typing import List
from OneDg.OD_grid_utils import get_sensory_inputs
from mainpt2 import test_data
from model.model import memorize
from OneDg.OD_grid import OneDGrid
from helpers.main_helpers import generate_path, apply_corruption, mean_cosine_similarity
from M4_benchmarks import smape, mase, split_into_train_test
import csv

# Configuration parameters
grid_types = '1D'
file = '/home/mcb/users/kduran1/temporal_torch/te'
output_dir = 'results'
w = 6  # Parameter for sensory input
lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
os.makedirs(output_dir, exist_ok=True)

# Load data and apply transformations
sensory_input = get_sensory_inputs(file, w)
#N = len(x_train)
x_train, y_train, x_test, y_test = split_into_train_test(sensory_input,700,48)
N = len(x_train)
print(len(x_test),len(x_train),len(y_train),len(y_test))
print(x_test[1],x_test[2],x_test[3])
print(y_test[1],y_test[2],y_test[3])




def iterative_forecast(model, x_test, fh):
    """
    Generate forecasts iteratively for the forecasting horizon.

    Args:
        model: The trained model.
        x_test: Input test data.
        fh: Forecasting horizon.

    Returns:
        y_hat_test: Tensor of forecasts for the next `fh` steps.
    """
    y_hat_test = []
    last_prediction = model.recall(x_test).item()  # First prediction from x_test

    for i in range(fh):
        y_hat_test.append(last_prediction)
        x_test = torch.roll(x_test, shifts=-1, dims=1)  # Shift x_test left by 1 timestep
        x_test[:, -1, :] = last_prediction  # Update the last timestep with the prediction
        last_prediction = model.recall(x_test).item()  # Generate the next prediction

    return torch.tensor(y_hat_test)


def run_experiment(x_train, y_train, x_test, y_test, lambdas, N_h, fh):
    """
    Runs the experiment using the provided train and test data.
    Args:
        x_train: Training input data.
        y_train: Training target data.
        x_test: Test input data.
        y_test: Test target data.
        lambdas: List of lambda values for the grid.
        N_h: Number of hidden units.
        fh: Forecasting horizon.
    Returns:
        model: The trained model.
        y_hat_test: Forecasts for the next `fh` steps.
    """
    grid = OneDGrid(lambdas)
    movements = generate_path(len(x_train))
    model = memorize(x_train, lambdas, N_h, grid, movements)
    # Iterative Forecasting
    y_hat_test = []
    x_test_iter = x_test.clone()  # Avoid modifying the original x_test

    for i in range(fh):
        # Generate the recall output
        recall_output = model.recall(x_test_iter)
        print(f"Step {i + 1}/{fh}:")
        print("Recall output shape:", recall_output.shape)  # Debugging
        print("Recall output:", recall_output)

        y_hat_test.append(recall_output)
        x_test_iter = torch.roll(x_test_iter, shifts=-1, dims=1)  # Roll along the time dimension
        if recall_output.dim() == 2 and recall_output.shape[1] == x_test_iter.shape[2]:
            # Update the last time step (column) for all batches
            x_test_iter[:, -1, :] = recall_output
        elif recall_output.dim() == 1:
            # Handle recall_output as a 1D tensor
            x_test_iter[:, -1, :] = recall_output.unsqueeze(1)  # Expand dimension to match

    y_hat_test = torch.stack(y_hat_test)  # Stack into a single tensor
    return model, y_hat_test


def generate_path(N: int) -> List[int]:
    """Generates a simple path with incremental steps."""
    return [i for i in range(N)]  # Movements with step size of 1

def detrend(insample_data):
    # Ensure insample_data is a 1D tensor
    if insample_data.dim() != 1:
        raise ValueError("Expected `insample_data` to be a 1D tensor.")

    # Create x values as indices
    x = torch.arange(len(insample_data), dtype=torch.float32)

    # Create the polynomial design matrix for a linear fit (degree 1)
    X = torch.stack([x, torch.ones_like(x)], dim=1)  # Stack x and intercept column

    # Ensure X and insample_data are compatible
    if X.dim() != 2 or X.size(0) != insample_data.size(0):
        raise ValueError("Mismatch between the dimensions of X and insample_data.")

    # Solve for coefficients (slope and intercept) using least squares
    solution = torch.linalg.lstsq(X, insample_data.unsqueeze(1))
    coefficients = solution.solution.squeeze()  # Extract the coefficients

    # Extract slope and intercept
    a, b = coefficients[0].item(), coefficients[1].item()
    return a, b

# if __name__ == '__main__':
#     # Run the experiment
#     model, grid, recalled_inputs = run_experiment(total_path_length=N, N_h=1000, corruption_level='none')
#
#     # Calculate mean cosine similarity between original and recalled inputs
#     mean_cos = mean_cosine_similarity(sensory_input, recalled_inputs)
#     print(f'Mean Cosine Similarity: {mean_cos}')
#
#     # Ensure `recalled_inputs` is a tensor for sMAPE and MASE calculations
#     recalled_inputs = torch.stack([torch.tensor(x, dtype=torch.float32) for x in recalled_inputs])
#
#     # sMAPE calculation using test data
#     smape_value = smape(y_test, recalled_inputs)
#     print(f"sMAPE: {smape_value}")
#
#     # MASE calculation
#     # Use `train_data` as `insample`, `test_data` as `y_test`, and `recalled_inputs[train_size:]` as `y_hat_test`
#     freq = 1  # Set frequency based on your data; 1 for no seasonality
#     mase_value = mase(x_train, y_test, recalled_inputs, freq)
#     print(f"MASE: {mase_value}")
#


if __name__ == '__main__':
    fh = 48  # Forecasting horizon
    N_h = 1000  # Hidden units
    lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    # Ensure detrended data for training
    x_train_detrended = []
    x_test_detrended = []
    trends_test = []

    # Detrend training data
    for seq in x_train:
        seq = seq.squeeze() if seq.dim() > 1 else seq
        a, b = detrend(seq)
        x = torch.arange(len(seq), dtype=torch.float32)
        trend = a * x + b
        x_train_detrended.append(seq - trend)

    # Detrend test data
    for seq in x_test:
        seq = seq.squeeze() if seq.dim() > 1 else seq
        a, b = detrend(seq)
        x = torch.arange(len(seq), dtype=torch.float32)
        trend = a * x + b
        x_test_detrended.append(seq - trend)
        trends_test.append((a, b))

    x_train_detrended = torch.stack(x_train_detrended)
    x_test_detrended = torch.stack(x_test_detrended)

    # Run experiment with iterative forecasting
    model, y_hat_test = run_experiment(
        x_train=x_train_detrended,
        y_train=y_train,
        x_test=x_test_detrended,
        y_test=y_test,
        lambdas=lambdas,
        N_h=N_h,
        fh=fh
    )
    # Reapply trends to predictions
    y_hat_test_with_trend = []
    for seq, (a, b) in zip(y_hat_test, trends_test):
        x = torch.arange(len(seq), dtype=torch.float32)
        trend = a * x + b
        y_hat_test_with_trend.append(seq + trend)
    y_hat_test_with_trend = torch.tensor(y_hat_test_with_trend)
    # Metrics Calculation
    smape_value = smape(y_test, y_hat_test_with_trend)
    print(f"sMAPE: {smape_value}")
    mase_value = mase(x_train.squeeze(1), y_test, y_hat_test_with_trend, freq=1)
    print(f"MASE: {mase_value}")
