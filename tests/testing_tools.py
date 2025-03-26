import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
#from tqdm import tqdm
from model.model import memorize
from OnedGrid.OneD_grid_utils import get_sensory_inputs, make_sequence
#from helpers.main_helpers import apply_corruption, generate_path

def get_c(p: float) -> torch.Tensor:
    """
    Compute the c parameter using the inverse error function.

    Args:
        p: A float value between 0 and 1.

    Returns:
        The c parameter as a tensor.
    """
    return torch.erfinv(2 * torch.tensor(p) - 1) * torch.sqrt(torch.tensor(2.0))

# def first_test():
#     mnist_dataset, labels = load_mnist()
#
#     input_vectors = mnist_dataset.reshape(mnist_dataset.shape[0], -1)
#
#     sensory_input = input_vectors[:2]
#     sensory_input = sensory_input / torch.max(sensory_input)
#
#     # Display the first 10 MNIST samples using Torchvision's grid
#     show_samples(mnist_dataset[:10], labels[:10])
#
#     movements = [(3 * i, i * 2) for i in range(10)]
#     lambdas = [3, 5, 7]
#     grid = (lambdas)
#
#     # Train the memory model
#     memory = memorize(sensory_input, lambdas, 10000, grid, movements,
#                       c=generate_c_from_Np([30], 10000)[0], var=1)
#
#     noisy_sensory_input = sensory_input + torch.rand_like(sensory_input) * 0.5 - 0.25
#     show_samples(noisy_sensory_input, labels[:2])
#
#     recalled_input = memory.recall(noisy_sensory_input)
#     show_samples(recalled_input, labels[:2])

def show_samples(images: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Display a grid of images with their labels.

    Args:
        images: A tensor containing images in shape (N, C, H, W).
        labels: A tensor containing labels corresponding to the images.
    """
    grid = make_grid(images.unsqueeze(1), nrow=10)
    print(f"Labels: {labels.tolist()}")
    print(grid)  # Print the tensor for display, modify as needed

def generate_c_from_p(p_list):
    return [get_c(p) for p in p_list]

def generate_c_from_Np(Np_list, Nh):
    return [get_c(Np / Nh) for Np in Np_list]

# def CIFAR_c_hyperparameter_search(variable_list, N, test='p', N_h=10000, var=1):
#     lambdas = [1, 2, 3]
#     square_grid = SquareGrid(lambdas)
#     hexagon_grid = HexagonGridModules(lambdas)

    transformed_data = get_sensory_inputs("/home/mcb/users/kduran1/temporal_torch/te")
    noisy_data = apply_corruption(transformed_data, 'low')
    movements = generate_path('levy', 10000, N)

    results = {}

    for grid_type in [square_grid, hexagon_grid]:
        results[str(grid_type)] = []
        print(f"Processing for {grid_type}...")

        with tqdm(total=len(variable_list), desc=f"Progress for {grid_type}") as pbar:
            for variable in variable_list:
                if test == 'NP':
                    c = generate_c_from_Np([variable], N_h)[0]
                else:
                    c = generate_c_from_p([variable])[0]

                memory = memorize(transformed_data, lambdas, N_h, grid_type, movements, c=c, var=var)
                recalled_input = memory.recall(noisy_data)
                cosine_scores = calculate_cosine_similarities(transformed_data, recalled_input)

                results[str(grid_type)].append({
                    'c': variable,
                    'mean_cosine_similarity': torch.mean(torch.tensor(cosine_scores)),
                    'std_cosine_similarity': torch.std(torch.tensor(cosine_scores))
                })

                pbar.update(1)

    return results

def calculate_cosine_similarities(list1, list2):
    similarities = []
    for i in range(len(list1)):
        vec1 = list1[i].flatten()
        vec2 = list2[i].flatten()
        similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
        similarities.append(similarity)
    return similarities

