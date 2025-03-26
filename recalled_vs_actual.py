import torch
import numpy as np
from typing import List, Tuple
from OneDg.OD_grid_utils import get_sensory_inputs
from model.model_pseudo_inv import memorize, Memory
from OneDg.OD_grid import OneDGrid  # Assuming this is for 1D grid
from helpers.dataset_helpers import transform_data, transform_single_tensor
# from helpers.plot_graphs_helpers import calculate_cosine_similarities
import csv  # For saving human-readable CSV files
from M4_benchmarks import  split_into_train_test
torch.set_printoptions(sci_mode=False)
torch.manual_seed(1)

#Configuration parameters
grid_types = '1D'
# file = 'dataset/hourly_data'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/daily_data'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/weekly_data'
#file = 'dataset/monthly_data'
# file = 'dataset/yearly_data'
file = 'dataset/toy_data'
output_dir = 'results'  # Changed from 'minigraphs' to 'results'


def generate_path(N: int) -> List[int]:
    """Generates a simple path with incremental steps."""
    movements = []
    for i in range(N): # Movements with step size of 1
        movements.append(i)
    return movements


def storage_test(norm_inputs: List[torch.Tensor], N_h: int, total_path_length: int, 
                 output_file: str = 'input_recall_pairs.csv'):
    """
    Store the normalized memories, and compute the average cosine similarity between the normalized memories and the
    recalled memories.
    We also return a percentage of how many correct grid states were recalled
    """
    grid = OneDGrid(lambdas)
    movements = generate_path(total_path_length)
    grids, model = memorize(norm_inputs, lambdas, N_h, grid, movements)
    cosine_similarity_list = []
    num_correct_grid_recall = 0
    cos = torch.nn.CosineSimilarity(dim=0)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['input', 'recalled_input'])

        for i, input in enumerate(norm_inputs):
            recalled_input, grid_input = model.recall([input])
            recalled_input = recalled_input[0].squeeze(0)
            if torch.equal(grid_input[0], grids[i]):
                num_correct_grid_recall += 1
            cosine_similarity_list.append(cos(input, recalled_input).item())

            # Write to line in csv
            written_input = [round(x, 3) for x in input.tolist()]
            written_recall = [round(x, 3) for x in recalled_input.tolist()]
            writer.writerow([written_input, written_recall])
    
    avg_cos_similarity = np.array(cosine_similarity_list).mean()
    return (avg_cos_similarity, (num_correct_grid_recall / float(len(grids))))

###################################################
###################################################
#######    M4 benchmarks     ######################
###################################################
###################################################
w = 6
# N_h = 1480
N_h = 35
# N_h = 6450
lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109]
#lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263]
sensory_input = get_sensory_inputs(file, w)

#total_path_length = len(sensory_input)#N

# Load data and apply transformations
#N = len(x_train)

# Configuration parameters
num_runs = 10
#hourly
fh=48
in_num = 700
#daily
# fh=14
# in_num = 93
# #weekly
# fh=13
# in_num = 80
# #monthly
# fh=18
# in_num = 42
# #yearly
# fh= 6
# in_num = 13
# Run the experiments and evaluate
x_train, y_train, x_test, y_test = split_into_train_test(sensory_input,in_num,fh)
N = len(x_train)
#norm_inputs = transform_data(sensory_input)
# norm_inputs  = transform_data(x_train.squeeze())
norm_inputs = transform_data(x_train.squeeze())
total_path_length = N

filename = 'input_recall_comparison.csv'
avg_cos_sim, grid_acc = storage_test(norm_inputs, N_h, total_path_length, filename)
print(f"Average Cosine Similarity: {avg_cos_sim}")
print(f"Grid Recall Accuracy: {grid_acc}")