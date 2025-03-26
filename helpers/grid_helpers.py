import torch
from typing import List


def module_split(g: torch.Tensor, lambdas: List[float], grid_type: str) -> List[torch.Tensor]:
    """
    Splits an input tensor into modularized grids based on specified sizes.

    :param g: A 1D torch tensor of data to be split into modules.
    :param lambdas: A list or array of integers where each integer represents the size of each module.
    :param grid_type: A string indicating whether the grid contains Hexagon Grid Cells or 1D Grid Cells.
    :return: A list of torch tensors, where each sub-tensor corresponds to a module of size specified in `lambdas`.
    """
    modules = []
    index = 0
    for size in lambdas:
        if grid_type == "hexagon":
            module_size = 3 * (size ** 2)
        elif grid_type == "1D":
            module_size = size
        else:
            module_size = size ** 2

        modules.append(g[index:index + module_size])
        index += module_size

    return modules


def softmax_with_beta(x: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Computes the softmax of a tensor with a given beta factor. **Used for the 1D denoising.

    :param x: The input tensor.
    :param beta: The scaling factor (temperature parameter).
    :return: Softmax probabilities of the input tensor.
    """
    # Scale the input tensor by the beta factor
    scaled_x = beta * x
    # Compute the exponentials, ensuring numerical stability by subtracting the max value
    exp_scaled = torch.exp(scaled_x - torch.max(scaled_x))

    # Compute the softmax probabilities
    softmax_values = exp_scaled / torch.sum(exp_scaled)

    return softmax_values

def denoising_power(noisy_g, b):
    """
    Calculates denoised grid.
    Args:
        b: denoising power : float
        noisy_g: noisy grid cell : tensor

    Returns:
        g_denoised : tensor
    """
    g_max = torch.max(noisy_g)
    g_stable = noisy_g / g_max
    g_denoised = torch.pow(g_stable, b)
    return g_denoised

# grid = torch.tensor([2.0, 0.1, 7.2]) ---> tensor([0.2778, 0.0139, 1.0000])
# b = 1.0



def denoise_g(noisy_g: torch.Tensor, lambdas: List[float], grid_type: str, denoising_factor: float) -> torch.Tensor:
    """
    Denoises an input tensor by selecting the nearest neighbor in each module.

    This function first splits the noisy grid tensor into smaller modules based on the sizes provided in `lambdas`.
    Each module is then processed to determine the nearest neighbor by finding the maximum value within the module.

    :param noisy_g: The noisy grid tensor that needs to be denoised.
    :param lambdas: A list containing the sizes of each grid module.
    :param grid_type: The type of grid cells being used (square, hexagon, or 1D).
    :param denoising_factor: Scaling factor for softmax-based denoising.
    :return: A denoised grid tensor of the same shape as noisy_g.
    """
    grid_modules = module_split(noisy_g, lambdas, grid_type)
    denoised_modules = []

    for n in grid_modules:
        if denoising_factor != -1.0:
            if torch.any(n != 0):  # Check if there are any non-zero elements in the module
                denoised_module = denoising_power(n, denoising_factor)
            else:
                # If the module is all zeros, create a tensor of the same shape with all zeros
                denoised_module = torch.zeros_like(n)
        else:
            if torch.any(n != 0):  # Check if there are any non-zero elements in the module
                max_value = torch.max(n)  # Find the maximum value in the module
                max_positions = (n == max_value).nonzero(as_tuple=True)[0]
                random_position = torch.randint(0, len(max_positions), (1,))
                denoised_module = torch.zeros_like(n)
                denoised_module[max_positions[random_position]] = 1
            else:
                # If the module is all zeros, create a tensor of the same shape with all zeros
                denoised_module = torch.zeros_like(n)

        denoised_modules.append(denoised_module)

    # Concatenate all modules into a single 1D tensor
    result_tensor = torch.cat(denoised_modules)

    return result_tensor


def inverse_denoising_power(encoded_g: torch.Tensor, b: float, original_max: float) -> torch.Tensor:
    """
    Reverses the power-based encoding to decode the original grid values.

    Args:
        encoded_g: The tensor that was encoded with denoising power.
        b: The power factor used in encoding.
        original_max: The original maximum value from the input before encoding.

    Returns:
        Decoded tensor with original scaling restored.
    """
    g_stable = torch.pow(encoded_g, 1 / b)  # Undo the power operation
    decoded_g = g_stable * original_max  # Rescale back to the original max value
    return decoded_g

def get_grid_index(grid:List[torch.tensor], module: torch.Tensor):
    for index, tensor in enumerate(grid):
        if torch.equal(module, tensor):
            return index
    return -1  # Return -1 if no match is found

def get_module_from_index(grid:List[torch.tensor],index: int):
    module = grid[index]
    return module

