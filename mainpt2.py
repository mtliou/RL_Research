import os
import torch
from collections import defaultdict
from typing import List, Tuple
from OneDg.OD_grid_utils import get_sensory_inputs
from model.model import memorize
from OneDg.OD_grid import OneDGrid  # Assuming this is for 1D grid
from helpers.main_helpers import apply_corruption, mean_cosine_similarity
from helpers.grid_helpers import get_grid_index, get_module_from_index
from helpers.dataset_helpers import transform_data
# from helpers.plot_graphs_helpers import calculate_cosine_similarities
import csv  # For saving human-readable CSV files
from M4_benchmarks import smape, mase
torch.set_printoptions(sci_mode=False)

# Configuration parameters
grid_types = '1D'
file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/te'
output_dir = 'results'  # Changed from 'minigraphs' to 'results'
w = 6  # Parameter for sensory input
# lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,151,157,163,
#167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263]
lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109]
# #101,103,107,109
N_h = 1480
# N_h = 6450
#print(sum(lambdas))

# lambdas = [
# 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
# 73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,
# 167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,
# 263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,
# 367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,
# 463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,
# 587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,
# 683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,
# 811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,
# 929,937,941,947,953,967,971,977]
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


sensory_input = get_sensory_inputs(file, w)
# sensory_input, mean, std = transform_data(sensory_input)
noisy_input = transform_data(sensory_input)

# # Load data and slice to the first 35 elements
# sensory_input = get_sensory_inputs(file, w)[:35]
# noisy_input = transform_data(sensory_input[:35])


N = len(sensory_input)
# print(N)
# # N_h=6450
def run_experiment(total_path_length: int, N_h: int, corruption_level: str):
    """Runs a single experiment for given parameters and saves results."""
    avg_recall = defaultdict(list)
    grid = OneDGrid(lambdas)  # Example grid configuration
        # Apply corruption to sensory input
    noisy_sensory_input = apply_corruption([tensor.clone() for tensor in sensory_input], corruption_level)
        # Generate movement paths
    movements = generate_path(N)
        # Store memories
    model = memorize(noisy_input, lambdas, N_h, grid, movements)
        # Recall memories with noisy input
    recalled_input = model.recall(noisy_input)
        # Calculate cosine similarities
    # similarities = calculate_cosine_similarities(sensory_input, recalled_input)
    # avg_recall[grid.N_g].extend(similarities)
    # # Save results to tensor and CSV files
    # save_results(avg_recall, total_path_length, N_h, corruption_level)
    return model, grid

def save_results(avg_recall, total_path_length, N_h, corruption_level):
    """Saves recall results with mean and standard deviation to both tensor and CSV files."""
    avg_recall_mean = {ng: torch.mean(torch.tensor(recall_list)).item() for ng, recall_list in avg_recall.items()}
    avg_recall_std = {ng: torch.std(torch.tensor(recall_list)).item() for ng, recall_list in avg_recall.items()}

    # Prepare a tensor with grid sizes, means, and standard deviations
    results_tensor = torch.tensor([
        [ng, avg_recall_mean[ng], avg_recall_std[ng]] for ng in avg_recall_mean
    ])

    # Save results to a CSV file for easy viewing
    csv_filename = f'corruption_{corruption_level}_path_line_length_{total_path_length}_Nh_{N_h}.csv'
    csv_filepath = os.path.join(output_dir, csv_filename)

    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Grid Size (N_g)', 'Mean Recall', 'Standard Deviation'])
        for ng in avg_recall_mean:
            writer.writerow([ng, avg_recall_mean[ng], avg_recall_std[ng]])

    print(f"CSV results saved to {csv_filepath}")

def generate_path(N: int) -> List[tuple[int,int]]:
    """Generates a simple path with incremental steps."""
    movements = []
    for i in range(N): # Movements with step size of 1
        movements.append((i,i))
    return movements


# def theo_capacity_test(num_memories, N_h, np_list, w, lambdas, percentages, file):
#     results = defaultdict(list)
#
#     for np_value in np_list:
#         print(f"Processing np={np_value}")
#         for mem, percent in zip(num_memories, percentages):
#             N= mem
#             # Simulate getting sensory inputs (replace this with actual function to load data)
#             sensory_input = get_sensory_inputs(file, w)[:mem]  # Load inputs up to current memory count
#             noisy_input = transform_data(sensory_input[:mem])  # Add noise or transformation
#             movements = generate_path(N)
#             # Initialize grid and model
#             grid = OneDGrid(lambdas)
#             model = memorize(noisy_input, lambdas, N_h, grid, movements, projection_dim=np_value)  # Example grid-cell-based model initialization
#             memorize(noisy_input, lambdas, N_h, grid, movements)
#             # Recall memories with noisy input
#             recalled_input = model.recall(noisy_input)
#
#             # Calculate cosine similarities
#             mean_cos = mean_cosine_similarity(noisy_input, recalled_input)  # Replace with your own function
#
#             # Store results
#             results[np_value].append({
#                 "percentage": percent,
#                 "num_memories": mem,
#                 "cosine_similarity": mean_cos.item() if isinstance(mean_cos, torch.Tensor) else mean_cos
#             })
#
#     # Save results in CSV format for external plotting
#     with open('results.csv', mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["np", "percentage", "num_memories", "cosine_similarity"])
#         for np_value, entries in results.items():
#             for entry in entries:
#                 writer.writerow([np_value, entry["percentage"], entry["num_memories"], entry["cosine_similarity"]])
#
#     print("Results saved to results.csv")
#     return results
# def theo_capacity_test(num_memories, N_h, np, w, lambdas, percentages):
#     for np_value in np:
#         for mem in num_memories:
#             sensory_input = get_sensory_inputs(file, w)[:mem]
#             noisy_input = transform_data(sensory_input[:mem])
#             avg_recall = defaultdict(list)
#             grid = OneDGrid(lambdas)  # Example grid configuration
#             # Recall memories with noisy input
#             recalled_input = model.recall(noisy_input)
#             # Calculate cosine similarities
#             mean_cos = mean_cosine_similarity(noisy_input,recalled_input)
#
#             return model, grid


if __name__ == '__main__':
    # Run the experiment
    model, grid = run_experiment(total_path_length=N, N_h=1480, corruption_level='none')
    # Create noisy input for testing recall


    # Recall the inputs from the noisy input
    recalled_inputs,grid_rec= model.recall(noisy_input)
    # Compare original and recalled inputs
    for original, recalled in zip(noisy_input, recalled_inputs):
        print(f"Original: {original}")
        print(f"Recalled: {recalled}\n")
    #
    # mean_cos = mean_cosine_similarity(noisy_input,recalled_inputs)
    # print(f'mean cosine similarity:{mean_cos}')
    # recalled_inputs = torch.stack([torch.tensor(x, dtype=torch.float32) for x in recalled_inputs])
    #
    #
    #
    # # # sMAPE calculation
    # # print(type(sensory_input))
    # # print(type(recalled_inputs))
    # # smape_value = smape(sensory_input,recalled_inputs)
    # #
    # # print(f"sMAPE: {smape_value}")
    # # mase_value = mase(sensory_input[:-48],sensory_input, recalled_inputs, 1)
    # # print(f"mase: {mase_value}")
    # percentages = [1, 3, 10, 20, 33, 50, 75, 90, 100, 110, 150, 200]
    # num_memories = [16, 50, 166, 333, 550, 833, 1250, 1500, 1666, 1833, 2500, 3300]
    # percentages = [1, 3, 10, 20, 33, 50, 75, 90, 100, 110,120,130,140]
    # num_memories = [26, 78, 260, 520, 858, 1951, 2341, 2601, 2861, 3121, 3381, 3642]
    # np = [5, 10, 20, 50, 100, 1000]
    # lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    # # N_h = 100
    # N_h = 121
    # w = 6
    # results = theo_capacity_test(num_memories, N_h, np, w, lambdas, percentages, file)
