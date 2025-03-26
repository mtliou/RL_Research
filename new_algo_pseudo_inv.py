import os
import torch
from collections import defaultdict
from typing import List, Tuple
from OneDg.OD_grid_utils import get_sensory_inputs
from model.model_pseudo_inv import memorize, Memory
from OneDg.OD_grid import OneDGrid  # Assuming this is for 1D grid
from helpers.main_helpers import apply_corruption, mean_cosine_similarity
from helpers.grid_helpers import get_grid_index, get_module_from_index
from helpers.dataset_helpers import transform_data, transform_single_tensor
# from helpers.plot_graphs_helpers import calculate_cosine_similarities
import csv  # For saving human-readable CSV files
from M4_benchmarks import split_into_train_test

torch.set_printoptions(sci_mode=False)
import random

random.seed(1)

# Configuration parameters
grid_types = '1D'
# file = 'dataset/hourly_data'
file = 'dataset/daily_data'
# file = 'dataset/weekly_data'
# file = 'dataset/monthly_data'
# file = 'dataset/yearly_data'
# file = 'dataset/toy_data'

#file = 'dataset/monthly_data'

output_dir = 'results'  # Changed from 'minigraphs' to 'results'


# w = 6
# N_h = 1480
# lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109]
# sensory_input = get_sensory_inputs(file, w)
# noisy_input = transform_data(sensory_input)
# total_path_length = len(sensory_input)#N

def generate_path(N: int) -> List[int]:
    """Generates a simple path with incremental steps."""
    movements = []
    for i in range(N):  # Movements with step size of 1
        movements.append(i)
    return movements


def compute_shifted_values(normalized_values, old_mean, old_std, normalized_removed, normalized_first):
    # Number of elements
    N = len(normalized_values)
    # Compute the removed element in normalized space (given)
    removed_element_normalized = normalized_removed
    new_element_normalized = normalized_first
    # Compute new mean in normalized space
    delta_mean_normalized = (new_element_normalized - removed_element_normalized) / N
    new_mean_normalized = 0 + delta_mean_normalized  # Normalized mean is 0
    # Compute new variance in normalized space
    delta_variance_normalized = (
                                        (new_element_normalized - 0) ** 2 - (removed_element_normalized - 0) ** 2
                                ) / N
    new_variance_normalized = 1 + delta_variance_normalized
    new_std_normalized = torch.sqrt(torch.tensor(new_variance_normalized))

    # Convert normalized space mean and std to real space
    new_mean_real = old_mean + new_mean_normalized * old_std
    new_std_real = old_std * new_std_normalized.item()

    # Recover real values from normalized values
    real_values = [(x * new_std_real + new_mean_real) for x in normalized_values]

    return {
        "new_mean_real": new_mean_real,
        "new_std_real": new_std_real,
        "real_values": real_values,
    }


def run_experiment(total_path_length: int, N_h: int, fh: int, first_input: torch.Tensor):
    """
    Total_path_length: number of memories to store
    N_h: number of place cells
    fh: forecast horizon, number of predictions to make
    Runs a single experiment for given parameters.
    """
    predictions_trail = []
    predictions_no_trail = []
    grid = OneDGrid(lambdas)  # Example grid configuration
    # Generate movement paths
    movements = generate_path(total_path_length)
    # Store memories
    grids, model = memorize(norm_inputs, lambdas, N_h, grid, movements)
    for i in range(fh):
        x_t, mean, std = transform_single_tensor(first_input)
        # print(f'test: {y_test[i]}')
        first_input_as_list = [x_t]
        trail_for_next = x_t[:-1]
        # og_in = [first_input]
        og_trail = first_input[:-1]
        # Recall memories
        recalled_input, grid_input = model.recall(first_input_as_list)
        # Move by one
        next_grid = grid.move2(grid_input[0])
        # Recall the next memory
        next_memory = Memory.get_memory_from_grid(model, next_grid)
        # Adjusting for pseudo-inverse
        next_memory = next_memory.squeeze(0)
        # Concatenate the recall with the trail of the original input
        d_t = torch.cat((next_memory[0].unsqueeze(0), trail_for_next), dim=0)

        # Update mean and std for the shifted dataset
        normalized_removed = (first_input[-1] - mean) / std
        normalized_first = next_memory[0]
        new_stats = compute_shifted_values(
            normalized_values=d_t.tolist(),
            old_mean=mean,
            old_std=std,
            normalized_removed=normalized_removed,
            normalized_first=normalized_first.item()
        )
        new_mean = new_stats["new_mean_real"]
        new_std = new_stats["new_std_real"]
        # Scale the next prediction
        x_tt = d_t * (new_std + 1e-7) + new_mean
        # Ensure og_trail is a tensor
        og_trail_tensor = torch.tensor(og_trail, device=x_tt.device)

        # Replace the last elements of x_tt with og_trail
        x_tt[-5:] = og_trail_tensor
        predictions_trail.append(x_tt)
        predictions_no_trail.append(x_tt[0])
        first_input = x_tt
        # print(f'recall:{x_tt}')
    print(y_test)
    print(predictions_trail)
    return predictions_trail


def multiscale_experiment(total_path_length: int, N_h: int, fh: int, first_input: torch.Tensor):
    """
    Total_path_length: number of memories to store
    N_h: number of place cells
    fh: forecast horizon, number of predictions to make
    Runs a single experiment for given parameters.
    """
    # predictions_trail = []
    # predictions_no_trail = []
    preds_trail = [[] for _ in range(len(model_probs))]
    preds_no_trail = [[] for _ in range(len(model_probs))]
    delta_preds_trail = []  # Stores our multiscale predictions
    grid_list = []
    # Init a grid for each model
    for _ in range(len(model_probs)):
        grid_list.append(OneDGrid(lambdas))  # Example grid configuration
    # Generate movement paths
    movements = generate_path(total_path_length)
    # Store memories for each model
    model_list = []
    for i, grid in enumerate(grid_list):
        grid_li, mod = memorize(norm_inputs[i], lambdas, N_h, grid, movements, model_probs[i])
        model_list.append(mod)

    for i in range(1, fh + 1):
        x_t, mean, std = transform_single_tensor(first_input)
        # print(f'test: {y_test[i]}')
        first_input_as_list = [x_t]
        trail_for_next = x_t[:-1]
        # og_in = [first_input]
        og_trail_tensor = first_input[:-1]
        # Initialize yhat as first input head
        yhat_t_plus_one = first_input[0].item()
        # Store the numerators for linear interpolation
        li_counters = [1] * len(grid_list)

        for j, model in enumerate(model_list):
            # If we are obtaining a new prediction from the model (if p=1 then always do)
            if i % (1 / model_probs[j]) == 1 or model_probs[j] == 1:
                # Recall memories
                recalled_input, grid_input = model.recall(first_input_as_list)
                # Move to the next grid state
                next_grid = grid.move2(grid_input[0].squeeze())
                # Recall the next memory
                next_memory = Memory.get_memory_from_grid(model, next_grid)
                flat_mem = (next_memory.view(-1))[0].unsqueeze(0)
                # Concatenate the recall with the trail of the original input
                d_t = torch.cat((flat_mem, trail_for_next), dim=0)
                # Update mean and std for the shifted dataset
                normalized_removed = (first_input[-1].item() - mean) / std
                normalized_first = next_memory[0]
                normalized_first = next_memory[0][0]
                new_stats = compute_shifted_values(
                    normalized_values=d_t.tolist(),
                    old_mean=mean,
                    old_std=std,
                    normalized_removed=normalized_removed,
                    normalized_first=normalized_first.item()
                )
                new_mean = new_stats["new_mean_real"]
                new_std = new_stats["new_std_real"]
                # Scale the next prediction
                x_tt = d_t * (new_std + 1e-7) + new_mean
                # Ensure og_trail is a tensor
                # og_trail_tensor = torch.tensor(og_trail, device=x_tt.device)

                # Replace the last elements of x_tt with og_trail
                x_tt[-5:] = og_trail_tensor
                preds_trail[j].append(x_tt)
                preds_no_trail[j].append(x_tt[0].item())
                # We can use 1 in numerator since we checked modulus in if condition
                li_coeff = 1 / (1 / model_probs[j])  # Get the LI coefficient
                new_pred = x_tt[0].item()  # Get the head of our prediction
                prev_pred = first_input[0].item()
                delta_change = new_pred - prev_pred  # Calculate delta change
                yhat_t_plus_one += li_coeff * delta_change  # Update yhat

                print(f"Delta Change: {li_coeff * delta_change}")
                print(f"New yhat: {yhat_t_plus_one}")

            # If we are not shifting the grid, just use previous shifted memory
            else:
                x_tt = preds_trail[j][-1]
                li_coeff = li_counters[j] / (1 / model_probs[j])
                # Reset the linear interpolation numerator once we reach a LI coeff of 1
                if li_counters[j] == (1 / model_probs[j]):
                    li_counters[j] = 1  # Reset to 1
                new_pred = x_tt[0].item()  # Get the LI coefficient
                prev_pred = first_input[0].item()
                delta_change = new_pred - prev_pred  # Calculate delta change
                yhat_t_plus_one += li_coeff * delta_change  # Update yhat

                print(f"Delta Change: {li_coeff * delta_change}")
                print(f"New yhat: {yhat_t_plus_one}")

            li_counters[j] += 1  # Increment the linear interpolation numerator by 1

        print(f"Run {i}")
        print(og_trail_tensor)
        print(torch.tensor([yhat_t_plus_one]))

        delta_prediction = torch.cat((torch.tensor([yhat_t_plus_one]), og_trail_tensor),
                                     0)  # Concat og trail to get our multiscale prediction
        print(delta_prediction)

        delta_preds_trail.append(delta_prediction)
        first_input = delta_prediction  # Update first input to our prediction for the next prediction
        # print(f'recall:{x_tt}')
    print(y_test)
    print(delta_preds_trail)
    return delta_preds_trail


def smape(actual, predicted):
    # Convert inputs to tensors if they are lists
    if isinstance(actual, list):
        actual = torch.cat(actual) if isinstance(actual[0], torch.Tensor) else torch.tensor(actual, dtype=torch.float32)
    if isinstance(predicted, list):
        predicted = torch.cat(predicted) if isinstance(predicted[0], torch.Tensor) else torch.tensor(predicted,
                                                                                                     dtype=torch.float32)
    # Flatten tensors if needed
    actual = actual.view(-1)
    print(predicted)
    predicted = predicted.view(-1)

    # Calculate sMAPE
    epsilon = 1e-8
    numerator = 2.0 * torch.abs(actual - predicted)
    denominator = torch.abs(actual) + torch.abs(predicted) + epsilon
    return torch.mean(numerator / denominator).item()


def mase(insample, y_test, y_hat_test, freq):
    # Ensure insample is a tensor, insample= number of data seen before the forecast
    if not isinstance(insample, torch.Tensor):
        insample = torch.tensor(insample, dtype=torch.float32)

    # Convert y_test and y_hat_test to tensors
    if isinstance(y_test, list):
        y_test = torch.stack(y_test) if isinstance(y_test[0], torch.Tensor) else torch.tensor(y_test,
                                                                                              dtype=torch.float32)
    if isinstance(y_hat_test, list):
        y_hat_test = torch.stack(y_hat_test) if isinstance(y_hat_test[0], torch.Tensor) else torch.tensor(y_hat_test,
                                                                                                          dtype=torch.float32)

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


###################################################
###################################################
#######    M4 benchmarks     ######################
###################################################
###################################################
w = 6
# N_h = 1480
N_h = 6450
#N_h = 10000
lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
           107, 109]
# lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263]
sensory_input = get_sensory_inputs(file, w)
# New probabilities
#model_probs = [1, 0.85, 0.60, 0.30, 0.25, 0.06, 0.03]
model_probs = [1, 0.5, 0.1]


# total_path_length = len(sensory_input)#N
# Load data and apply transformations
# N = len(x_train)


# total_path_length = N
#
# y_hat = run_experiment(N,N_h,48, x_test[-1].squeeze())

# recall_values= [tensor.tolist() for tensor in y_hat]
# print(recall_values)
# smape_value = smape(y_test, y_hat)
# print(f"sMAPE: {smape_value}")
# mase_value = mase(sensory_input, y_test, y_hat, freq=1)
# #mase_value = mase(x_train.squeeze(), y_test, y_hat, freq=1)
# print(f"MASE: {mase_value}")



def run_experiments_and_evaluate(total_path_length, N_h, fh, first_input, num_runs=10):
    smape_scores = []
    mase_scores = []

    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")

        # Run the experiment
        y_hat = run_experiment(total_path_length, N_h, fh, first_input)
        print(y_test)
        print(y_hat)
        # Calculate sMAPE
        smape_value = smape(y_test, y_hat)
        print(smape_value)
        smape_scores.append(smape_value)

        # Calculate MASE
        mase_value = mase(x_test.squeeze(), y_test, y_hat, freq=1)
        print(mase_value)
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


def run_experiments_and_evaluate_multiscale(total_path_length, N_h, fh, first_input, num_runs=10):
    smape_scores = []
    mase_scores = []

    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")

        # Run the experiment
        y_hat = multiscale_experiment(total_path_length, N_h, fh, first_input)
        print(y_test)
        print(y_hat)
        # Calculate sMAPE
        smape_value = smape(y_test, y_hat)
        print(smape_value)
        smape_scores.append(smape_value)

        # Calculate MASE
        mase_value = mase(x_test.squeeze(), y_test, y_hat, freq=1)
        print(mase_value)
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
# #daily
fh=14
in_num = 93
# weekly
# fh=13
# in_num = 80
# #monthly
# fh = 18
# in_num = 42
# #yearly
# fh= 6
# in_num = 13
# Run the experiments and evaluate
x_train, y_train, x_test, y_test = split_into_train_test(sensory_input, in_num, fh)


# print(f'size xtrain:{len(x_train)},size ytrain:{len(y_train)}, size xtest:{len(x_test)}, size ytest:{len(y_test)}')
# N = len(x_train)
# norm_inputs  = transform_data(sensory_input)

def augment_dataset(dset: List[torch.tensor], shift_probs: List[float]):
    '''
    Take the x_train set, and create augmented datasets based on the shifting probabiilities of each model.
    This allows us to store the data regularly.
    '''
    x_train_list = [[] for _ in range(len(shift_probs))]
    # Iterate through probabilities for each model
    for i, p in enumerate(shift_probs):
        # Store the head of the last stored datapoint
        prev_data_head = dset[0][:5]
        # Iterate through dataset
        for j, data in enumerate(dset):
            # Always add the first datapoint for simplicity
            if j == 0:
                x_train_list[i].append(data)
                continue

            # Probabilistically add the data to its models dataset
            if random.random() < p:
                # We augment the data to preserve the fact that datapoints are being skipped
                aug_data = torch.cat((data[0].unsqueeze(0), prev_data_head), 0)
                x_train_list[i].append(aug_data)
                # Update the next datapoint's tail as we have stored a datapoint
                prev_data_head = aug_data[:5]

    return x_train_list


x_train_list = augment_dataset(x_train.squeeze(), model_probs)

# For regular model
norm_inputs = transform_data(x_train.squeeze())

# For multiscale model
# norm_inputs = []
# for i, dset in enumerate(x_train_list):
#     print(f"Amount of Datapoints Stored for Probability {model_probs[i]} Model: {len(dset)}")
#     norm_inputs.append(transform_data(dset))

N = len(norm_inputs[0])
results = run_experiments_and_evaluate(total_path_length=N, N_h=N_h, fh=fh, first_input=x_test[-1].squeeze(), num_runs=num_runs)
#results = multiscale_experiment(total_path_length=N, N_h=N_h, fh=fh, first_input=x_test[-1].squeeze())
#results = run_experiments_and_evaluate_multiscale(total_path_length=N, N_h=N_h, fh=fh, first_input=x_test[-1].squeeze())
# Print the results
print(f"sMAPE Mean: {results['sMAPE_mean']:.4f}, sMAPE Std: {results['sMAPE_std']:.4f}")
print(f"MASE Mean: {results['MASE_mean']:.4f}, MASE Std: {results['MASE_std']:.4f}")













