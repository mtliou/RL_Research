import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from M4_benchmarks import  split_into_train_test
from helpers.dataset_helpers import transform_data, transform_single_tensor
from OneDg.OD_grid_utils import get_sensory_inputs
torch.manual_seed(1)

# Comment out import and prediction method being used to plot
from prediction_one import run_experiment
pred_method = 'Incremental_Mean+Var'

# from prediction_two import run_experiment
# pred_method = 'Incremental_LR'

# from prediction_3 import run_experiment
# pred_method = 'Batch_Mean+Var'

# from prediction_four import run_experiment
# pred_method = 'Batch_LR'

y_axis_limit = 200

# Comment out data file being used
#file = 'dataset/hourly_data'
#file = 'dataset/daily_data'
#file = 'dataset/weekly_data'
#file = 'dataset/monthly_data'
#file = 'dataset/yearly_data'
file = 'dataset/toy_data'

buffer_len = 6

#N_h = 1480
#N_h = 2960
N_h = 3700

lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109]
#lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263]
sensory_input = get_sensory_inputs(file, buffer_len)

num_runs = 10
# hourly
fh=48
in_num = 700

# daily
# fh=14
# in_num = 93

# weekly
# fh=13
# in_num = 80

# monthly
# fh=18
# in_num = 42

# yearly
# fh= 6
# in_num = 13

x_train, y_train, x_test, y_test = split_into_train_test(sensory_input,in_num,fh)
norm_inputs  = transform_data(x_train.squeeze())
N = len(norm_inputs)


def run_multiple_experiments(total_path_length, N_h, fh, num_runs=10):
    """
    Run the multiscale experiment multiple times and collect results.
    
    Args:
        total_path_length: Path length for the experiment
        N_h: Number of place cells
        fh: Forecast horizon
        num_runs: Number of times to run the experiment
    
    Returns:
        all_yhats: List of lists (runs × predictions)
    """
    all_yhats = []
    
    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")
        yhats_trail = run_experiment(
            total_path_length=total_path_length, 
            N_h=N_h, 
            fh=fh, 
            first_input=x_test[-1].squeeze(),
            norm_inputs=norm_inputs,
            y_test=y_test
        )
        yhats = [yhat_trail[0].item() for yhat_trail in yhats_trail]
        all_yhats.append(yhats)
    
    return all_yhats

def plot_multiple_predictions(training_values, test_values_actual, all_yhats, output_file_prefix):
    """
    Plot multiple prediction runs with individual predictions as faint lines and average as bold.
    
    Args:
        training_values: Array of training data points
        test_values_actual: Array of actual test values
        all_yhats: List of lists (runs × predictions)
        output_file_prefix: Prefix for the output file name
    """
    # Directory to store plots to (create if dir doesn't exist)
    subdir = 'num_hpc_cell_plots'
    os.makedirs(subdir, exist_ok=True)

    # Non-zoomed plot
    plt.figure(figsize=(12, 6))

    # Number each entry instead of using dates
    training_indices = range(1, len(training_values) + 1)
    test_indices = range(len(training_values) + 1, len(training_values) + len(test_values_actual) + 1)

    # Plot the last 1/6 of the training data, excluding the testing section
    sixth_index = len(training_values) * 5 // 6
    plt.plot(training_indices[sixth_index:], training_values[sixth_index:], label="Training Data", color="blue")

    # Plot test data with their actual indices
    plt.plot(test_indices, test_values_actual, label="Test Data", color="orange", linestyle='--')

    # Plot each prediction run as a faint line
    for i, yhats in enumerate(all_yhats):
        # Convert tensor predictions to numerical values if needed
        predictions = [y.item() if isinstance(y, torch.Tensor) else y for y in yhats]
        plt.plot(test_indices[:len(predictions)], predictions, color="green", alpha=0.2, linewidth=0.8,
                label="Individual Predictions" if i == 0 else None)  # Only add to legend once

    # Calculate and plot the average prediction as a bold line
    avg_predictions = []
    for step in range(len(all_yhats[0])):
        step_vals = [run[step] if isinstance(run[step], (int, float)) else run[step].item() for run in all_yhats]
        avg_predictions.append(sum(step_vals) / len(step_vals))
    
    plt.plot(test_indices[:len(avg_predictions)], avg_predictions, color="green", linewidth=1.5,
             label="Average Prediction")

    # Add a shaded region for the test data
    plt.axvspan(test_indices[0], test_indices[-1], color="gray", alpha=0.2, label="Test Period")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim(0, y_axis_limit)
    plt.title(f"{output_file_prefix}_non_zoomed")
    plt.legend()
    plt.grid(True)
    filename = f"{output_file_prefix}_non_zoomed.png"
    filepath = os.path.join(subdir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    # Zoomed-in plot
    plt.figure(figsize=(12, 6))

    # Plot test data with their actual indices
    plt.plot(test_indices, test_values_actual, label="Test Data", color="orange", linestyle='--')

    # Plot each prediction run as a faint line
    for i, yhats in enumerate(all_yhats):
        predictions = [y.item() if isinstance(y, torch.Tensor) else y for y in yhats]
        plt.plot(test_indices[:len(predictions)], predictions, color="green", alpha=0.2, linewidth=0.8,
                label="Individual Predictions" if i == 0 else None)

    # Plot the average prediction as a bold line
    plt.plot(test_indices[:len(avg_predictions)], avg_predictions, color="green", linewidth=1.5,
             label="Average Prediction")

    # Add a shaded region for the test data
    plt.axvspan(test_indices[0], test_indices[-1], color="gray", alpha=0.2, label="Test Period")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim(0, y_axis_limit)
    plt.title(f"{output_file_prefix}_zoomed")
    plt.legend()
    plt.grid(True)
    filename = f"{output_file_prefix}_zoomed.png"
    filepath = os.path.join(subdir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


data = pd.read_csv(file, header=None, names=["Value"])
# Separate training and testing sets
training_values = data["Value"][:-fh].values  # All but the last fh values for training
test_values_actual = data["Value"][-fh:].values  # Last fh values for testing

all_yhats = run_multiple_experiments(total_path_length=N, N_h=N_h, fh=fh, num_runs=10)

# Call the function after running your experiments
plot_multiple_predictions(training_values, test_values_actual, all_yhats, f"{pred_method}_Method_{N_h}_hpc_cells")