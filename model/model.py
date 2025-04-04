import torch
from typing import List, Tuple
#from six import moves
from torchvision.transforms.v2.functional import normalize
from OneDg.OD_grid import OneDGrid
from helpers.grid_helpers import denoise_g, get_grid_index, get_module_from_index
from OneDg.OD_grid_utils import get_stats
from helpers.grid_helpers import denoising_power, inverse_denoising_power
from helpers.layer_computation_helpers import (
    compute_p_from_g, compute_p_from_s, compute_g_from_p, compute_s_from_p, incremental_update
)

class Memory:
    def __init__(self, N_h: int, lambdas: List[int], input_size: int, grid: OneDGrid, projection_dim: int = 20,
                 var: float = 1.0, b=1000.0) -> None:
        """
        Initializes the Memory model with hippocampus cells, grid sizes, and input size.
        """
        # k = len(lambdas)  # number of grid modules
        self.grid_size = grid_size = grid.N_g
        self.N_h = N_h
        self.projection_dim = projection_dim
        number_of_grid_modules = grid.N_modules
        c = self._find_c(projection_dim, N_h, number_of_grid_modules, var)

        # Initialize weights
        self.Wpg = torch.normal(mean=c / number_of_grid_modules,
                                std=(var / number_of_grid_modules) ** 0.5,
                                size=(N_h, grid_size))
        self.Wgp = torch.zeros(grid_size, N_h)
        self.Wsp = torch.zeros(input_size, N_h)
        self.Wps = torch.zeros(N_h, input_size)
        self.lambdas = lambdas
        self.grid = grid
        self.b = b  # Denoising power
        self.mean = 0
        self.std = 0

        # Store scaling factors for decoding
        self.input_max_values = []

    def _find_c(self, projection_dim, N_h, number_of_grid_modules, var):
        c = -(2 * var) ** 0.5 * torch.special.erfinv(torch.tensor(1 - 2 * projection_dim / N_h))
        return float(c)

    def normalize_s(self, s):
        self.std, self.mean = get_stats()
        if isinstance(s, list):
            # Normalize each tensor in the list individually
            normalized_s = [(tensor - self.mean) / self.std for tensor in s]
        else:
            # Normalize the single tensor
            normalized_s = (s - self.mean) / self.std
        return normalized_s

    def store_memory(self, s: torch.Tensor, num_iterations=1) -> None:
        """
        Stores a memory after encoding the input vector.
        """
        #already normalized
        s_encoded = s
        #s_encoded = self.normalize_s(s)
        for _ in range(num_iterations):
            p = compute_p_from_g(self.grid.g, self.Wpg)
            self.Wgp = incremental_update(self.Wgp, p, self.grid.g)
            self.Wsp = incremental_update(self.Wsp, p, s_encoded)
            self.Wps = incremental_update(self.Wps, s_encoded, p)

    def learn_path(self, input_memories: List[torch.Tensor], movements: List[int]) -> None:
        """
        Learns a sequence of memories along a path.
        """
        for m, v in zip(input_memories, movements):
            self.grid.move()
            self.store_memory(m)

    def recall(self, noisy_sensory_input: List[torch.Tensor]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Recalls the original inputs from noisy sensory inputs (normalized before being passed).
        """
        recalled_inputs = []
        grid_recalled = []

        #need the good grid module to be able top recover the tensor value (with std and mean)
        for model_input in noisy_sensory_input:
            noisy_p = compute_p_from_s(model_input, self.Wps)
            noisy_g = compute_g_from_p(noisy_p, self.Wgp)
            denoised_g = denoise_g(noisy_g, self.lambdas, self.grid.type, self.b)
            denoised_p = compute_p_from_g(denoised_g, self.Wpg)
            denoised_s_encoded = compute_s_from_p(denoised_p, self.Wsp)

            #decoded_s = (denoised_s_encoded * self.std) + self.mean
            recalled_inputs.append(denoised_s_encoded)
            grid_recalled.append(denoised_g)
            #decoded_s = denoised_s_encoded
        return recalled_inputs, grid_recalled

    def get_memory_from_grid(self, grid_module):
        denoised_g = grid_module
        denoised_p = compute_p_from_g(denoised_g, self.Wpg)
        denoised_s_encoded = compute_s_from_p(denoised_p, self.Wsp)
        return denoised_s_encoded

    def get_many_memory_for_grid(self, grid_module_list):
        memories = []
        for grid_module in grid_module_list:
            denoised_g = grid_module
            denoised_p = compute_p_from_g(denoised_g, self.Wpg)
            denoised_s_encoded = compute_s_from_p(denoised_p, self.Wsp)
            memories.append(denoised_s_encoded)
        return memories

def memorize(
    sensory_input: List[torch.Tensor], lambdas: List[int], N_h: int, grid_type: OneDGrid,
    movements: List[int], projection_dim=20, var=1.0, b=1000.0
) -> Memory:
    """
    Trains the hippocampus-grid weights with the given sensory input.
    """
    input_size = sensory_input[0].shape[0]
    memory = Memory(N_h, lambdas, input_size, grid_type, projection_dim, var, b)

    # Learn path and store memories
    memory.learn_path(sensory_input, movements)
    return memory
