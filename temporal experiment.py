import os
import torch
from collections import defaultdict
from typing import List, Tuple
from OneDg.OD_grid_utils import get_sensory_inputs
from model.model import memorize
from model.model import Memory
from OneDg.OD_grid import OneDGrid # Assuming this is for 1D grid
from helpers.main_helpers import apply_corruption, mean_cosine_similarity
from helpers.dataset_helpers import transform_data, denormalize_data, transform_single_tensor
# from helpers.plot_graphs_helpers import calculate_cosine_similarities
import csv  # For saving human-readable CSV files
from M4_benchmarks import smape, mase
from helpers.grid_helpers import get_grid_index, get_module_from_index
torch.set_printoptions(sci_mode=False)

# Configuration parameters
grid_types = '1D'
file = '/home/mcb/users/kduran1/temporal_torch/te'
output_dir = 'results'  # Changed from 'minigraphs' to 'results'
w = 6 # Parameter for sensory input
lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
N_h=1000
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
# Load data and apply transformations
sensory_input = get_sensory_inputs(file, w)
N = len(sensory_input)

#NORMALIZE each tensor locally and store their respective mean and std
normalized_sensory_input,means,stds = transform_data(sensory_input)
#print(normalized_sensory_input,means,stds)

def generate_path(N: int) -> List[int]:
    """Generates a simple path with incremental steps."""
    return [i for i in range(N)]  # Movements with step size of 1

def run_experiment(total_path_length: int, N_h: int, corruption_level: str, input: List[torch.tensor], fh: int):
    """Given a noisy tensor input, it denoises it to recover the grid module,
    then recalls the next fh(forecast horizon) following memories"""
    grid = OneDGrid(lambdas)  # Example grid configuration
    recalls = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #                  STORING                                 #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # Generate movement paths
    movements = generate_path(N)
        # Store the normalized sensory inputs, and their mean and std
    model = memorize(normalized_sensory_input, lambdas, N_h, grid, movements)
    #store the means and std for each modules
    model.mean = means
    model.std = stds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #                  RECALL                                  #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Recall memories with noisy input, starting by normalizing it
    normalized_input,m,s = transform_data(input)
    #get the noisy normalized input
    #NEED TO VERIFY THE RECALL FUNCTION
    recalled_input, denoised_g_list = model.recall(normalized_input)
    #get the only recalled input from the list as a tensor
    denoised_g = denoised_g_list[0]
    recall = recalled_input[0]
    for i in range(fh):
        next_module = OneDGrid.move2(denoised_g)
        #make a memory recall function that recalls given a grid module
        next_memory = Memory.get_memory_from_grid(next_module)
        recalls.append(next_memory)
        denoised_g = next_module
    return model, grid

# inputs = [
#     torch.tensor([20.7000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
#     torch.tensor([17.9000, 20.7000, 0.0000, 0.0000, 0.0000, 0.0000])
# ]
#
# model, grid = run_experiment(N, N_h, 'none', inputs)
input = torch.tensor([20.0,0.0,0.0,0.0,0.0,0.0])
model, grid = run_experiment(N,N_h, 'none', input, 48)