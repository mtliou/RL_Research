import os
import torch
from collections import defaultdict
from typing import List, Tuple
from OneDg.OD_grid_utils import get_stats
from model.model import memorize
from OneDg.OD_grid import OneDGrid
from helpers.main_helpers import apply_corruption
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as F


# Configuration parameters
output_dir = 'results'
image_output_dir = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/mnist_images'
os.makedirs(image_output_dir, exist_ok=True)

lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
           53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
N_h = 1000

# Load MNIST data and flatten images to 1D vectors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten each 28x28 image to a 784-length vector
])

mnist_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

# Take a sample of MNIST images as sensory input
mnist_images, _ = next(iter(mnist_loader))  # Get a batch of images
mnist_images = [img for img in mnist_images]  # Convert batch to list of images
N = len(mnist_images)


def generate_path(N: int) -> List[int]:
    """Generates a simple path with incremental steps."""
    return [i for i in range(N)]  # Movements with step size of 1


def run_experiment(total_path_length: int, N_h: int, corruption_level: str):
    """Runs a single experiment for given parameters and saves results."""
    avg_recall = defaultdict(list)
    grid = OneDGrid(lambdas)  # Initialize grid

    # Apply corruption to sensory input
    noisy_sensory_input = apply_corruption([tensor.clone() for tensor in mnist_images], corruption_level)

    # Generate movement paths
    movements = generate_path(N)

    model = None  # Ensure model is initialized

    for i in range(2, 30):  # Fix loop range
        model = memorize(mnist_images, lambdas, N_h, grid, movements, projection_dim=i)

    if model is None:
        raise ValueError("Model was not initialized. Check the loop range in run_experiment().")

    # Recall memories with noisy input
    recalled_input = model.recall(noisy_sensory_input)

    if isinstance(recalled_input, list):
        recalled_input = torch.stack(recalled_input)  # Convert list to tensor

    return model, grid, mnist_images, recalled_input



def save_side_by_side_images(original_images, recalled_images, filename="side_by_side_comparison_np5_b1000.png"):
    """Saves a side-by-side comparison of original and recalled images."""
    # Convert tensors to PIL images
    pil_originals = [F.to_pil_image(img.view(28, 28)) for img in original_images]
    pil_recalled = [F.to_pil_image(img.view(28, 28)) if isinstance(img, torch.Tensor) else F.to_pil_image(torch.tensor(img).view(28, 28)) for img in recalled_images]


    # Concatenate images horizontally (original next to recalled)
    side_by_side_images = []
    for original, recalled in zip(pil_originals, pil_recalled):
        combined = Image.new('L', (original.width * 2, original.height))
        combined.paste(original, (0, 0))
        combined.paste(recalled, (original.width, 0))
        side_by_side_images.append(combined)

    # Stack images vertically
    total_height = sum(img.height for img in side_by_side_images)
    combined_image = Image.new('L', (side_by_side_images[0].width, total_height))
    y_offset = 0
    for img in side_by_side_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the final combined image
    output_path = os.path.join(image_output_dir, filename)
    combined_image.save(output_path)
    print(f"Side-by-side comparison image saved to {output_path}")


if __name__ == '__main__':
    # Run the experiment with the MNIST dataset
    model, grid, original_images, recalled_images = run_experiment(total_path_length=N, N_h=1000,
                                                                   corruption_level='none')

    # Save a side-by-side comparison of the first batch of original and recalled images
    print("Saving Side-by-Side Comparison of Original and Recalled Images to Disk:")
    save_side_by_side_images(original_images, recalled_images)

    # Calculate cosine similarity for each original and recalled image
    cosine_similarities = [torch.nn.functional.cosine_similarity(original, recalled, dim=0).item()
                           for original, recalled in zip(original_images, recalled_images)]
    avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)

    print(f"\nAverage Cosine Similarity: {avg_cosine_similarity:.4f}")
    for i, similarity in enumerate(cosine_similarities):
        print(f"Cosine Similarity for Image {i + 1}: {similarity:.4f}")
# import csv
# import os
# import torch
#
# from collections import defaultdict
# from typing import List, Tuple
#
# from model.model import memorize
# from OneDg.OD_grid import OneDGrid
# from helpers.main_helpers import apply_corruption, mean_cosine_similarity
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from PIL import Image
# import torchvision.transforms.functional as F
#
# # Configuration parameters
# output_dir = 'results'
# # lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,149,151,157,163,
# # 167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271]
# lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,127,131,137,139,149,151,157,163,167]
#
# #lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113]
# #N_h = 1460
# # N_h = 1580
# # N_h = 8850
# #
# N_h = 1371
# print(sum(lambdas))
# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)
#
# # Load MNIST data and flatten images to 1D vectors
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.view(-1))  # Flatten each 28x28 image to a 784-length vector
# ])
#
# mnist_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
# mnist_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
#
# # Take a sample of MNIST images as sensory input
# mnist_images, _ = next(iter(mnist_loader))  # Get a batch of images
# mnist_images = [img for img in mnist_images]  # Convert batch to list of images
# N = len(mnist_images)
# print(N)
#
# # def generate_path(N: int) -> List[int]:
# #     """Generates a simple path with incremental steps."""
# #     return [i for i in range(N)]  # Movements with step size of 1
# def generate_path(N: int) -> List[tuple[int,int]]:
#     """Generates a simple path with incremental steps."""
#     movements = []
#     for i in range(N): # Movements with step size of 1
#         movements.append((i,i))
#     return movements
#
# def run_experiment(total_path_length: int, N_h: int, corruption_level: str, projection_dim: int):
#     """Runs a single experiment for a given projection dimension and returns cosine similarities."""
#     grid = OneDGrid(lambdas)  # Initialize grid
#
#     # Apply corruption to sensory input
#     noisy_sensory_input = apply_corruption([tensor.clone() for tensor in mnist_images], corruption_level)
#
#     # Generate movement paths
#     movements = generate_path(N)
#
#     # Store memories
#     model = memorize(mnist_images, lambdas, N_h, grid, movements, projection_dim=projection_dim)
#     print((model.Wpg).shape)
#
#     # Recall memories with noisy input
#     recalled_input = model.recall(noisy_sensory_input)
#
#     # Calculate cosine similarity for each original and recalled image
#     cosine_similarities = [torch.nn.functional.cosine_similarity(original, recalled, dim=0).item()
#                            for original, recalled in zip(mnist_images, recalled_input)]
#
#     return cosine_similarities
#
#
#
# def theo_capacity_test(num_memories, N_h, np_list, lambdas, percentages, output_dir):
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Define the transform to flatten images
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.view(-1))  # Flatten each 28x28 image to a 784-length vector
#     ])
#
#     # Load MNIST dataset
#     # mnist_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
#     # mnist_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
#
#     # Initialize results dictionary
#     results = defaultdict(list)
#
#     # Iterate over the different np values
#     for np_value in np_list:
#         print(f"Processing np={np_value}")
#
#         for mem, percent in zip(num_memories, percentages):
#             N = mem
#             # Sample the required number of MNIST images
#             mnist_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
#             mnist_loader = DataLoader(mnist_dataset, batch_size=mem, shuffle=True)
#             mnist_images = []
#             for images, _ in mnist_loader:
#                 mnist_images.extend(images)  # Add images from batch to the list
#                 if len(mnist_images) >= mem:  # Stop once we have enough images
#                     mnist_images = mnist_images[:mem]
#                     break
#
#             # Convert sampled images to tensor
#             sensory_input = torch.stack(mnist_images)
#
#             # Add noise to the sensory input
#             noisy_input = sensory_input
#             movements = generate_path(N)
#             # Initialize grid and model
#             grid = OneDGrid(lambdas)
#             model = memorize(noisy_input, lambdas, N_h, grid, movements, projection_dim=np_value)
#
#             # Recall memories with noisy input
#             recalled_input = model.recall(noisy_input)
#
#             # Calculate cosine similarity
#             mean_cos = mean_cosine_similarity(noisy_input, recalled_input)
#
#             # Store results
#             results[np_value].append({
#                 "percentage": percent,
#                 "cosine_similarity": mean_cos.item() if isinstance(mean_cos, torch.Tensor) else mean_cos
#             })
#
#     # Save results in CSV format for external plotting
#     output_file = os.path.join(output_dir, 'results_mnist.csv')
#     with open(output_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["np", "percentage", "num_memories", "cosine_similarity"])
#         for np_value, entries in results.items():
#             for entry in entries:
#                 writer.writerow([np_value, entry["percentage"], entry["cosine_similarity"]])
#
#     print(f"Results saved to {output_file}")
#     return results
#
# if __name__ == '__main__':
#     # cosine_similarities = run_experiment(total_path_length=N, N_h=7080, corruption_level='none',projection_dim=20)
#     # avg_cosine_similarity = np.mean(cosine_similarities)
#     # print(cosine_similarities)
#     # print(avg_cosine_similarity)
#     # # Dictionary to store results for each projection dimension
#     # results = []
#     # best_np = None
#     # best_avg_cosine_similarity = -float('inf')
#     # best_std_dev = float('inf')
#     #
#     # # Run the experiment across different projection dimensions with increments of 2
#     # for proj_dim in [1, 2,10,20,50,100]:  # Increment by 2
#     #     print(f"Running experiment for projection_dim = {proj_dim}")
#     #
#     #     # Get cosine similarities for the current projection dimension
#     #     cosine_similarities = run_experiment(total_path_length=N, N_h=7080, corruption_level='none',
#     #                                          projection_dim=proj_dim)
#     #
#     #     # Calculate average and standard deviation
#     #     avg_cosine_similarity = np.mean(cosine_similarities)
#     #     std_dev_cosine_similarity = np.std(cosine_similarities)
#     #
#     #     # Store results as tuples
#     #     results.append((proj_dim, avg_cosine_similarity, std_dev_cosine_similarity))
#     #
#     #     # Check if this is the best `Np` based on highest cosine similarity and smallest std deviation
#     #     if avg_cosine_similarity > best_avg_cosine_similarity or (
#     #         avg_cosine_similarity == best_avg_cosine_similarity and std_dev_cosine_similarity < best_std_dev):
#     #         best_np = proj_dim
#     #         best_avg_cosine_similarity = avg_cosine_similarity
#     #         best_std_dev = std_dev_cosine_similarity
#     #
#     #     # Print the results for this projection dimension
#     #     print(
#     #         f"Projection Dim: {proj_dim} | Average Cosine Similarity: {avg_cosine_similarity:.4f} | Standard Deviation: {std_dev_cosine_similarity:.4f}")
#     #
#     # # Display the results in a table format
#     # print("\nFinal Results Table:")
#     # print(f"{'Projection Dim':<15}{'Average Cosine Similarity':<25}{'Standard Deviation':<20}")
#     # for proj_dim, avg_cosine, std_dev in results:
#     #     print(f"{proj_dim:<15}{avg_cosine:<25.4f}{std_dev:<20.4f}")
#     #
#     # # Print the best projection dimension
#     # print(f"\nBest Projection Dimension (Np) with Highest Average Cosine Similarity and Lowest Std Dev:")
#     # print(f"Projection Dim: {best_np}, Average Cosine Similarity: {best_avg_cosine_similarity:.4f}, "
#     #       f"Standard Deviation: {best_std_dev:.4f}")
#     #
#     # # Optionally, save the results to a text file
#     # output_path = os.path.join(output_dir, "cosine_similarity_results_c3.txt")
#     # with open(output_path, "w") as f:
#     #     f.write("Projection Dim\tAverage Cosine Similarity\tStandard Deviation\n")
#     #     for proj_dim, avg_cosine, std_dev in results:
#     #         f.write(f"{proj_dim}\t{avg_cosine:.4f}\t{std_dev:.4f}\n")
#     #     # f.write(f"\nBest Projection Dimension: {best_np}, Average Cosine Similarity: {best_avg_cosine_similarity:.4f}, "
#     #     #         f"Standard Deviation: {best_std_dev:.4f}\n")
#     # print(f"\nResults saved to {output_path}")
#     # percentages = [1, 3, 10, 20, 33, 50, 75, 100, 110, 150, 200]
#     # num_memories = [23, 71,239,479,791,1198,1798,2397, 2637,3596,4795]
#     percentages = [0.0001,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
#     num_memories = [2,11,23,58,108,216,583,1083,2166,5415]
#     np = [5, 10, 20, 50, 100, 1000]
#     lambdas = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167]
#     N_h = 2914

    # results = theo_capacity_test(num_memories, N_h, np, lambdas, percentages, 'mnist_results')
