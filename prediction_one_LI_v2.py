import os
import torch
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from OneDg.OD_grid_utils import get_sensory_inputs
from model.model import memorize, Memory
from OneDg.OD_grid import OneDGrid  # Assuming this is for 1D grid
from helpers.main_helpers import apply_corruption, mean_cosine_similarity
from helpers.grid_helpers import get_grid_index, get_module_from_index
from helpers.dataset_helpers import transform_data, transform_single_tensor
# from helpers.plot_graphs_helpers import calculate_cosine_similarities
import csv  # For saving human-readable CSV files
from M4_benchmarks import  split_into_train_test
torch.set_printoptions(sci_mode=False)
import random
random.seed(1)
torch.manual_seed(1)

#Configuration parameters
grid_types = '1D'
buffer_length = 6

#file = 'dataset/hourly_data'
#model_probs = [1, 0.6, 0.25]

#file = 'dataset/daily_data'
#model_probs = [1, 0.85, 0.03]

file = 'dataset/weekly_data'
model_probs = [1, 0.7, 0.3]

#file = 'dataset/monthly_data'
#model_probs = [1, 0.8, 0.06]

#file = 'dataset/yearly_data'
#model_probs = [1, 0.6, 0.25]


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
    for i in range(N): # Movements with step size of 1
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

def multiscale_experiment(total_path_length: int, N_h: int, fh: int, first_inputs: List[torch.Tensor]):
    """
    Total_path_length: number of memories to store
    N_h: number of place cells
    fh: forecast horizon, number of predictions to make
    first_inputs: list of first inputs for each model
    Runs a single experiment for given parameters.
    """

    preds_upper = [[] for _ in range(len(model_probs))]
    preds_lower = [[] for _ in range(len(model_probs))]
    delta_preds_trail = [] # Stores our multiscale predictions
    grid_list = []
    # Init a grid for each model
    for _ in range(len(model_probs)):
        grid_list.append(OneDGrid(lambdas))  # Example grid configuration
    # Generate movement paths
    movements = generate_path(total_path_length)
    # Store memories for each model
    model_list = []
    for i, grid in enumerate(grid_list):
        model_list.append(memorize(norm_inputs[i], lambdas, N_h, grid, movements, model_probs[i]))

    # Metric trackers
    deltas = [[] for _ in range(len(model_probs))]
    yhats = []

    # Store the numerators for linear interpolation
    upper_bds = [1 / p for p in model_probs]

    delta_change_scaled = 0

    for i in range(1, fh + 1):
        # Initialize yhat as first input head 
        if len(yhats) == 0:
            yhat_t_plus_one = first_inputs[0][0].item()
        else:
            yhat_t_plus_one = yhats[-1]

        # sum_fi = 0
        # for fi in first_inputs:
        #     sum_fi += fi[0].item()
        # yhat_t_plus_one = sum_fi / len(first_inputs)

        for j, model in enumerate(model_list):
            tskips = 1 / model_probs[j] # How many memories the model skips on average
            og_trail_tensor = first_inputs[j][:-1]

            # Note: only one condition can be true
            if i > upper_bds[j] or i == 1:            # CHECK IF i > upper_bound[j]

                # Get first input
                first_input = first_inputs[j]
                x_t, mean, std = transform_single_tensor(first_input)
                #print(f'test: {y_test[i]}')
                first_input_as_list = [x_t]
                trail_for_next = x_t[:-1]
                #og_in = [first_input]

                # if first condition was true, we need to update our upper bounds
                if i !=  1:
                    upper_bds[j] += 1 / model_probs[j]
                    x_tt_lower = preds_upper[j]

                # Recall memories
                recalled_input, grid_input = model.recall(first_input_as_list)
                # Move to the next grid state
                next_grid = grid.move2(grid_input[0])
                # Recall the next memory
                next_memory = Memory.get_memory_from_grid(model, next_grid)
                # Concatenate the recall with the trail of the original input
                d_t = torch.cat((next_memory[0].unsqueeze(0), trail_for_next), dim=0)

                # Update mean and std for the shifted dataset
                normalized_removed = (first_input[-1].item() - mean) / std
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
                # og_trail_tensor = torch.tensor(og_trail, device=x_tt.device)

                # Replace the last elements of x_tt with og_trail
                x_tt[-(buffer_length - 1):] = og_trail_tensor

                li_coeff = (upper_bds[j] - i) / tskips

                #delta_change_scaled = ((1 - li_coeff) * x_tt[0].item()) + (li_coeff * first_input[0].item())

                # Subtract previous delta change so we are not adding that delta change multiple times (no compunding of deltas)
                delta_change_scaled = ((1 - li_coeff) * (x_tt[0].item() - first_input[0].item())) - delta_change_scaled

                yhat_t_plus_one += delta_change_scaled / len(model_probs)

                # Update first inputs for next change in predictions
                first_inputs[j] = x_tt

                # Store the upper and lower deltas
                preds_upper[j].append(x_tt)
                preds_lower[j].append(first_input)

                # # We can use 1 in numerator since we checked modulus in if condition
                # li_coeff = 1 / (1 / model_probs[j]) # Get the LI coefficient
                # new_pred = x_tt[0].item() # Get the head of our prediction
                # prev_pred = first_input[0].item()
                # delta_change = new_pred - prev_pred # Calculate delta change
                # yhat_t_plus_one += li_coeff * delta_change # Update yhat

                # Metrics
                deltas[j].append(delta_change_scaled)

                # print(f"Delta Change: {li_coeff * delta_change}")
                # print(f"New yhat: {yhat_t_plus_one}")

            # If we are not shifting the grid, just use previous shifted memory
            else:
                x_tt = preds_upper[j][-1]
                x_tt_lower = preds_lower[j][-1]
                li_coeff = (upper_bds[j] - i) / tskips

                #delta_change_scaled = ((1 - li_coeff) * x_tt[0].item()) + (li_coeff * x_tt_lower[0].item())
                
                # Subtract previous delta change so we are not adding that delta change multiple times (no compunding of deltas)
                delta_change_scaled = ((1 - li_coeff) * (x_tt[0].item() - x_tt_lower[0].item())) - delta_change_scaled

                yhat_t_plus_one += delta_change_scaled / len(model_probs)

                # Metrics
                deltas[j].append(delta_change_scaled)

                # print(f"Delta Change: {li_coeff * delta_change}")
                # print(f"New yhat: {yhat_t_plus_one}")

            #first_inputs[j] = x_tt # Update first input to our prediction for the next prediction

        # print(f"Run {i}")
        # print(og_trail_tensor)
        # print(torch.tensor([yhat_t_plus_one]))

        # Metrics
        yhats.append(yhat_t_plus_one)

        delta_prediction = torch.cat((torch.tensor([yhat_t_plus_one]), og_trail_tensor), 0) # Concat og trail to get our multiscale prediction
        
        # print(delta_prediction)
        # first_inputs[j] = delta_prediction # Update first input to our prediction for the next prediction

        delta_preds_trail.append(delta_prediction)

    # print(y_test)
    # print(delta_preds_trail)
    # print(f"Length of predictions: {len(delta_preds_trail)}")
    # print(f"y_test length: {len(y_test)}")

    return delta_preds_trail, deltas, yhats

def run_multiple_experiments(total_path_length, N_h, fh, first_inputs, num_runs=10):
    """
    Run the multiscale experiment multiple times and collect results.
    
    Args:
        total_path_length: Path length for the experiment
        N_h: Number of place cells
        fh: Forecast horizon
        first_inpust: List of initial input tensors
        num_runs: Number of times to run the experiment
    
    Returns:
        all_deltas: List of lists of lists (runs × models × values)
        all_yhats: List of lists (runs × predictions)
    """
    all_deltas = []
    all_yhats = []
    all_results = []
    
    for run in range(num_runs):
        print(f"Running experiment {run + 1}/{num_runs}...")
        results, deltas, yhats = multiscale_experiment(
            total_path_length=total_path_length, 
            N_h=N_h, 
            fh=fh, 
            first_inputs=copy.deepcopy(first_inputs)
        )
        all_results.append(results)
        all_deltas.append(deltas)
        all_yhats.append(yhats)
    
    return all_results, all_deltas, all_yhats

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
    # Ensure insample is a tensor, insample= number of data seen before the forecast
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

###################################################
###################################################
#######    M4 benchmarks     ######################
###################################################
###################################################
w = 6
N_h = 1480
#N_h = 6450
lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109]
#lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263]
sensory_input = get_sensory_inputs(file, w)
#total_path_length = len(sensory_input)#N
# Load data and apply transformations
#N = len(x_train)


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

print(len(sensory_input))

# Configuration parameters
num_runs = 10
# #hourly
# fh=48
# in_num = 700
# #daily
# fh=14
# in_num = 93
#weekly
# fh=13
# in_num = 80
# #monthly
fh = 18
in_num = 42
# #yearly
# fh= 6
# in_num = 13
# Run the experiments and evaluate
x_train, y_train, x_test, y_test = split_into_train_test(sensory_input,in_num,fh)
#print(f'size xtrain:{len(x_train)},size ytrain:{len(y_train)}, size xtest:{len(x_test)}, size ytest:{len(y_test)}')
# N = len(x_train)
#norm_inputs  = transform_data(sensory_input)

def augment_dataset(dset: List[torch.Tensor], shift_probs: List[float]):
    '''
    Take the x_train set, and create augmented datasets based on the shifting probabiilities of each model.
    This allows us to store the data regularly.
    '''
    x_train_list = [[] for _ in range(len(shift_probs))]
    # Iterate through probabilities for each model
    for i, p in enumerate(shift_probs):
        # Store the head of the last stored datapoint
        prev_data_head = dset[0][:(buffer_length - 1)]
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
                prev_data_head = aug_data[:(buffer_length - 1)]
            
    return x_train_list

x_train_list = augment_dataset(x_train.squeeze(), model_probs)
            
norm_inputs = []
for i ,dset in enumerate(x_train_list):
    print(f"Amount of Datapoints Stored for Probability {model_probs[i]} Model: {len(dset)}")
    norm_inputs.append(transform_data(dset))

N = len(norm_inputs[0])
#results = run_experiments_and_evaluate(total_path_length=N, N_h=N_h, fh=fh, first_input=x_test[-1].squeeze(),
                          #             num_runs=num_runs)

# Create a list of first inputs for each model
def get_fi_list(first_input: torch.Tensor, dsets: List[List[torch.Tensor]]):
    '''
    Using the original first input, get the first input for each model depending on its previously stored memory
    '''
    fi_list = []
    for dset in dsets:
        # Add first element of og first input to the "head" of the last stored input
        new_first = torch.cat((first_input[0].unsqueeze(0), dset[-1][:(buffer_length - 1)]), 0)
        fi_list.append(new_first)
    return fi_list

fi_list = get_fi_list(x_test[-1].squeeze(), x_train_list)

#results, deltas_no_scale, deltas_scaled, yhats = multiscale_experiment(total_path_length=N, N_h=N_h, fh=fh, first_inputs=copy.deepcopy(fi_list))


###################################################
###################################################
#######    Plotting     ###########################
###################################################
###################################################


# print("deltas_no_scale:", deltas_no_scale)
# print("deltas_scaled:", deltas_scaled)

# def plot_deltas(deltas, title, filename):
#     """
#     Plot the deltas and save the figure.
    
#     Args:
#         deltas: List of lists containing delta values
#         title: Title of the plot
#         filename: Filename to save the plot
#     """
#     plt.figure(figsize=(10, 6))
    
#     # Plot each sublist as a separate line
#     for i, delta_list in enumerate(deltas):
#         plt.plot(delta_list, label=f"Model {i+1} (p={model_probs[i]})")
    
#     plt.title(title)
#     plt.xlabel("Forecast Step")
#     plt.ylabel("Delta Value")
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Save the figure
#     plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
#     plt.close()

# # Plot both delta graphs
# plot_deltas(deltas_no_scale, f"Raw Delta Changes by Model, Forecast Horizon of {fh}", "deltas_no_scale")
# plot_deltas(deltas_scaled, f"Scaled Delta Changes by Model, Forecast Horizon of {fh}", "deltas_scaled")

# print(f"Plots saved as 'deltas_no_scale.png' and 'deltas_scaled.png'")

# Print the results
# print(f"sMAPE Mean: {results['sMAPE_mean']:.4f}, sMAPE Std: {results['sMAPE_std']:.4f}")
# print(f"MASE Mean: {results['MASE_mean']:.4f}, MASE Std: {results['MASE_std']:.4f}")


data = pd.read_csv(file, header=None, names=["Value"])

# Separate training and testing sets
training_values = data["Value"][:-fh].values  # All but the last 48 values for training
test_values_actual = data["Value"][-fh:].values  # Last 48 values for testing

def plot_multiple_predictions(training_values, test_values_actual, all_yhats, output_file_prefix):
    """
    Plot multiple prediction runs with individual predictions as faint lines and average as bold.
    
    Args:
        training_values: Array of training data points
        test_values_actual: Array of actual test values
        all_yhats: List of lists (runs × predictions)
        output_file_prefix: Prefix for the output file name
    """
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
    plt.title("Training Data, Test Data and Predictions")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_file_prefix}_non_zoomed.png", dpi=300, bbox_inches='tight')
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
    plt.title("Zoomed View of Test Data and Predictions")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_file_prefix}_zoomed.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiple_deltas(all_deltas, title, filename, model_probs):
    """
    Plot multiple runs of deltas with faint lines for individual runs and bold lines for averages.
    
    Args:
        all_deltas: List of lists of lists (runs × models × values)
        title: Title of the plot
        filename: Filename to save the plot
        model_probs: List of model probabilities for the legend
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate average deltas for each model across runs
    num_models = len(all_deltas[0])
    
    # Dynamically generate colors based on the number of models
    colors = cm.rainbow(np.linspace(0, 1, num_models))
    
    # Prepare arrays for average calculations
    avg_deltas = [[] for _ in range(num_models)]
    
    # Plot individual runs as faint lines
    for run_idx, run_deltas in enumerate(all_deltas):
        for model_idx, delta_list in enumerate(run_deltas):
            plt.plot(delta_list, color=colors[model_idx], alpha=0.2, linewidth=0.8)
            
            # Collect values for average calculation
            if run_idx == 0:
                avg_deltas[model_idx] = [[val] for val in delta_list]
            else:
                for step_idx, val in enumerate(delta_list):
                    if step_idx < len(avg_deltas[model_idx]):
                        avg_deltas[model_idx][step_idx].append(val)
                    else:
                        avg_deltas[model_idx].append([val])
    
    # Calculate and plot average lines (bold)
    for model_idx, model_avgs in enumerate(avg_deltas):
        # Calculate average for each step
        avg_line = [sum(step_vals) / len(step_vals) for step_vals in model_avgs]
        plt.plot(avg_line, color=colors[model_idx], linewidth=1.5, 
                 label=f"Model {model_idx+1} (p={model_probs[model_idx]}) - Average")
    
    plt.title(title)
    plt.xlabel("Forecast Step")
    plt.ylabel("Delta Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Run multiple experiments and generate plots
num_runs = 10
all_results, all_deltas, all_yhats = run_multiple_experiments(
    total_path_length=N, 
    N_h=N_h, 
    fh=fh, 
    first_inputs=fi_list,
    num_runs=num_runs
)

# Plot with individual runs (faint) and averages (bold)
plot_multiple_deltas(all_deltas, 
                    f"Raw Delta Changes by Model, Forecast Horizon of {fh} ({num_runs} runs)", 
                    f"deltas_{fh}", 
                    model_probs)

# Call the function after running your experiments
plot_multiple_predictions(training_values, test_values_actual, all_yhats, f"{fh}_predictions")





