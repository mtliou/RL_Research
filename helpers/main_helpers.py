import torch
from typing import List, Tuple
from path_generators.regular_line import generate_points
from path_generators.brownian_walk import generate_and_plot
from path_generators.levy_flight import LevyWalk

def generate_path(path: str, path_len: int, N: int) -> List[Tuple[int, int]]:
    """
    Given a type of path and its length, generate a list of coordinates on that path.

    :param path: The type of path to generate.
    :param path_len: The length of the path.
    :param N: The maximum range or dimension of the path.
    :return: A list of coordinates on the path.
    """
    if path == 'straight':
        return generate_points(path_len, N)
    elif path == 'brownian':
        return generate_and_plot(path_len, N)
    elif path == 'levy':
        levy_walk = LevyWalk(alpha=1.5, beta=0, max_dist=path_len * 2)
        return levy_walk.generate_path(path_len, N)
    else:
        raise ValueError(f"Unknown path type: {path}")

def apply_corruption(images: List[torch.Tensor], corruption: str) -> List[torch.Tensor]:
    """
    Add noise to images based on the specified corruption level.

    :param images: A list of image tensors to corrupt.
    :param corruption: The corruption level ('none', 'low', 'medium', 'high').
    :return: A list of corrupted image tensors.
    """
    if corruption == 'none':
        return images
    elif corruption == 'low':
        noise = [img + torch.rand_like(img) * 0.2 - 0.1 for img in images]
    elif corruption == 'medium':
        noise = [img + torch.rand_like(img) * 0.5 - 0.25 for img in images]
    elif corruption == 'high':
        noise = [img + torch.rand_like(img) * 1.0 - 0.5 for img in images]
    else:
        raise ValueError(f"Unknown corruption level: {corruption}")

    # Ensure pixel values are still in a valid range (e.g., [0, 1])
    return [torch.clamp(img, 0.0, 1.0) for img in noise]


def mean_cosine_similarity(original_patterns, recalled_patterns):
    # Convert lists to tensors if necessary
    if isinstance(original_patterns, list):
        original_patterns = torch.stack(original_patterns)
    if isinstance(recalled_patterns, list):
        recalled_patterns = torch.stack(recalled_patterns)

    # Ensure the tensors are 2D: (batch_size, flattened pattern size)
    original_patterns_flat = original_patterns.view(original_patterns.size(0), -1)
    recalled_patterns_flat = recalled_patterns.view(recalled_patterns.size(0), -1)

    # Calculate cosine similarities for each pair in the batch
    dot_products = torch.sum(original_patterns_flat * recalled_patterns_flat, dim=1)
    original_norms = torch.norm(original_patterns_flat, dim=1)
    recalled_norms = torch.norm(recalled_patterns_flat, dim=1)

    # Calculate cosine similarity for each pair
    cosine_similarities = dot_products / (original_norms * recalled_norms)

    # Calculate the mean cosine similarity across all pairs
    mean_cosine_sim = torch.mean(cosine_similarities)
    return mean_cosine_sim.item()  # Return as a Python float

