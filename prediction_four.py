import os
import torch
from collections import defaultdict
from typing import List, Tuple
import pandas as pd

from torch.fx.experimental.unification.unification_tools import first

from OneDg.OD_grid_utils import get_sensory_inputs
from model.model import memorize, Memory
from OneDg.OD_grid import OneDGrid  # Assuming this is for 1D grid
from helpers.main_helpers import apply_corruption, mean_cosine_similarity
from helpers.grid_helpers import get_grid_index, get_module_from_index
from helpers.dataset_helpers import transform_data, transform_single_tensor
# from helpers.plot_graphs_helpers import calculate_cosine_similarities
import csv  # For saving human-readable CSV files
from M4_benchmarks import  split_into_train_test
# Define linear regression scaling
from torch.nn import Linear
from torch.optim import SGD
torch.set_printoptions(sci_mode=False)
torch.manual_seed(1)

#Configuration parameters
grid_types = '1D'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/hourly_data'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/daily_data'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/weekly_data'
file = 'dataset/monthly_data'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/yearly_data'
output_dir = 'results'  # Changed from 'minigraphs' to 'results'

def generate_path(N: int) -> List[int]:
    """Generates a simple path with incremental steps."""
    movements = []
    for i in range(N): # Movements with step size of 1
        movements.append(i)
    return movements


def smape(actual, predicted):
    # Convert inputs to tensors if they are lists
    if isinstance(actual, list):
        actual = torch.cat(actual) if isinstance(actual[0], torch.Tensor) else torch.tensor(actual, dtype=torch.float32)
    if isinstance(predicted, list):
        predicted = torch.cat(predicted) if isinstance(predicted[0], torch.Tensor) else torch.tensor(predicted, dtype=torch.float32)
    # Flatten tensors if needed
    actual = actual.view(-1)
    predicted = predicted.view(-1)

    # Calculate sMAPE
    epsilon = 1e-8
    numerator = 2.0 * torch.abs(actual - predicted)
    denominator = torch.abs(actual) + torch.abs(predicted) + epsilon
    return torch.mean(numerator / denominator).item()

def mase(insample, y_test, y_hat_test, freq):
    # Ensure insample is a tensor
    if not isinstance(insample, torch.Tensor):
        insample = torch.tensor(insample, dtype=torch.float32)

    # Convert y_test and y_hat_test to tensors
    if isinstance(y_test, list):
        y_test = torch.stack(y_test) if isinstance(y_test[0], torch.Tensor) else torch.tensor(y_test, dtype=torch.float32)
    if isinstance(y_hat_test, list):
        y_hat_test = torch.stack(y_hat_test) if isinstance(y_hat_test[0], torch.Tensor) else torch.tensor(y_hat_test, dtype=torch.float32)

    # Flatten tensors
    y_test = y_test.view(-1)
    y_hat_test = y_hat_test.view(-1)
    # Generate naive forecast
    y_hat_naive = insample[:-freq]
    y_true_naive = insample[freq:]

    # Calculate the denominator term for MASE
    masep = torch.mean(torch.abs(y_true_naive - y_hat_naive))

    # Calculate the MASE metric
    return torch.mean(torch.abs(y_test - y_hat_test)) / masep


def run_experiment(total_path_length: int, N_h: int, fh: int, first_input: torch.Tensor, norm_inputs: List[torch.Tensor], y_test: List[torch.Tensor]):
    """
    Total_path_length: number of memories to store
    N_h: number of place cells
    fh: forecast horizon, number of predictions to make
    Runs a single experiment for given parameters.
    """
    predictions =[]
    scaled_recall = []
    grid = OneDGrid(lambdas)  # Example grid configuration
        # Generate movement paths
    movements = generate_path(total_path_length)
        # Store memories
    model = memorize(norm_inputs , lambdas, N_h, grid, movements)
    # Initialize linear regression parameters manually
    b = torch.tensor(0.0, requires_grad=False)
    x_t, mean, std = transform_single_tensor(first_input)
    first_input_as_list = [x_t]
    #recall th eseries of memories from the first recall, iteratively

    recalled_input, grid_input = model.recall(first_input_as_list)
    forecast_grids = grid.move2_return(grid_input[0], fh)
    forecast_memories = Memory.get_many_memory_for_grid(model, forecast_grids)

    for i in range(fh):
            # Recall memories
        recalled_input, grid_input = model.recall(first_input_as_list)
        #move by the index of the memory to recall
        #first itteration
        if i==0:
            #trail_input = x_t[:-1]
            next_memory = forecast_memories[0]
            trail_d_t = next_memory[1:]  # Next memory trail
            target = first_input[:-1]
            trail_d_t = trail_d_t.unsqueeze(1)  # Convert from shape [5] to [5, 1]
            X_aug = torch.cat((trail_d_t, torch.ones(5, 1)), dim=1)
            reg_lambda = 0.0001  # Regularization strength
            theta = torch.pinverse(X_aug.T @ X_aug + reg_lambda * torch.eye(X_aug.shape[1])) @ X_aug.T @ target
            w = theta[:-1]  # The weight vector, treat w like the std
            b = theta[-1]  # The bias term, treat b like the mean
            xtt = (next_memory * w) + b
            print(f'y_test = {y_test[i]}')
            print(f"x_tt:{xtt}")
            # Store scaled prediction
            predictions.append(xtt)
        else:
            next_memory = forecast_memories[i]
            trail_d_t = next_memory[1:]  # Next memory trail
            target = (predictions[i-1])[:-1]
            trail_d_t = trail_d_t.unsqueeze(1)
            print(target)
            print(trail_d_t)
            X_aug = torch.cat((trail_d_t, torch.ones(5, 1)), dim=1)
            reg_lambda = 0.0001  # Regularization strength
            theta = torch.pinverse(X_aug.T @ X_aug + reg_lambda * torch.eye(X_aug.shape[1])) @ X_aug.T @ target
            w = theta[:-1]  # The weight vector, treat w like the std
            b = theta[-1]  # The bias term, treat b like the mean
            xtt = (next_memory * w) + b

            # Store scaled prediction
            predictions.append(xtt)


    print(y_test)
    print(predictions)
    return predictions

###################################################
###################################################
#######    M4 benchmarks     ######################
###################################################
###################################################
w = 6
N_h = 1480
#N_h = 6450
lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109]
#lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263]
sensory_input = get_sensory_inputs(file, w)

#total_path_length = len(sensory_input)#N

# Load data and apply transformations
#N = len(x_train)
def run_experiments_and_evaluate(total_path_length, N_h, fh, first_input, num_runs=10):
    smape_scores = []
    mase_scores = []

    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")

        # Run the experiment
        y_hat = run_experiment(total_path_length, N_h, fh, first_input)

        # Calculate sMAPE
        smape_value = smape(y_test, y_hat)
        smape_scores.append(smape_value)

        # Calculate MASE
        mase_value = mase(x_test.squeeze(), y_test, y_hat, freq=1)
        mase_scores.append(mase_value)

    # Convert scores to tensors
    smape_scores = torch.tensor(smape_scores, dtype=torch.float32)
    mase_scores = torch.tensor(mase_scores, dtype=torch.float32)

    # Compute statistics
    mean_smape = torch.mean(smape_scores).item()
    std_smape = torch.std(smape_scores, unbiased=True).item()  # unbiased=True for sample std dev
    mean_mase = torch.mean(mase_scores).item()
    std_mase = torch.std(mase_scores, unbiased=True).item()

    return {
        "sMAPE_mean": mean_smape,
        "sMAPE_std": std_smape,
        "MASE_mean": mean_mase,
        "MASE_std": std_mase
    }


# Configuration parameters
num_runs = 10

#hourly
# fh=48
# in_num = 700
#daily
# fh=14
# in_num = 93
# #weekly
# fh=13
# in_num = 80
# #monthly
fh=18
in_num = 42
# #yearly
# fh= 6
# in_num = 13
# Run the experiments and evaluate
x_train, y_train, x_test, y_test = split_into_train_test(sensory_input,in_num,fh)
print(f'size xtrain:{len(x_train)},size ytrain:{len(y_train)}, size xtest:{len(x_test)}, size ytest:{len(y_test)}')
N = len(x_train)
#norm_inputs = transform_data(sensory_input)
norm_inputs  = transform_data(x_train.squeeze())
total_path_length = N

#results = run_experiments_and_evaluate(total_path_length=N, N_h=N_h, fh=fh, first_input=x_test[-1].squeeze(),
       #                                num_runs=num_runs)
results = run_experiment(total_path_length=N, N_h=N_h, fh=fh, first_input=x_test[-1].squeeze(), norm_inputs=norm_inputs, y_test=y_test)
# Print the results
# print(f"sMAPE Mean: {results['sMAPE_mean']:.4f}, sMAPE Std: {results['sMAPE_std']:.4f}")
# print(f"MASE Mean: {results['MASE_mean']:.4f}, MASE Std: {results['MASE_std']:.4f}")
# y_hat = run_experiment(N,N_h,48, x_test[-1].squeeze())
#
# smape_value = smape(y_test, y_hat)
# print(f"sMAPE: {smape_value}")
# mase_value = mase(sensory_input, y_test, y_hat, freq=1)
# #mase_value = mase(x_train.squeeze(), y_test, y_hat, freq=1)
# print(f"MASE: {mase_value}")

