import torch
import plotly.express as px
import plotly.graph_objects as go
from typing import List

# Cosine Similarity Functions
def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor, epsilon: float = 1e-10) -> float:
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    norm1 = torch.norm(vec1) + epsilon
    norm2 = torch.norm(vec2) + epsilon
    dot_product = torch.dot(vec1 / norm1, vec2 / norm2)
    return torch.clamp(dot_product, -1.0, 1.0).item()

def calculate_cosine_similarities(list1: List[torch.Tensor], list2: List[torch.Tensor]) -> List[float]:
    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length.")
    return [cosine_similarity(list1[i], list2[i]) for i in range(len(list1))]

# Plotting Functions with Plotly
def plot_results(
    original: torch.Tensor, noisy: torch.Tensor, recall: torch.Tensor,
    cosine_sim: float, title1="Original", title2="Noisy", title3="Recall"
):
    fig = go.Figure()

    # Original Image
    fig.add_trace(go.Image(z=original.numpy(), name=title1))

    # Noisy Image
    fig.add_trace(go.Image(z=noisy.numpy(), name=title2))

    # Recalled Image
    fig.add_trace(go.Image(z=recall.numpy(), name=title3))

    fig.update_layout(
        title=f"Cosine Similarity: {cosine_sim:.3f}",
        showlegend=True,
        height=600,
        width=800
    )
    fig.show()

def plot_chart(
    title: str, originals: List[torch.Tensor], noises: List[torch.Tensor],
    results: List[List[torch.Tensor]], parameters: List[str], test='P'
):
    fig = go.Figure()
    similarities = [
        calculate_cosine_similarities(originals, results[i]) for i in range(len(results))
    ]

    for i, parameter in enumerate(parameters):
        avg_similarity = torch.tensor(similarities[i]).mean().item()

        # Plot a grid of images with average similarity for each parameter
        fig.add_trace(go.Scatter(
            x=[parameter] * len(originals),
            y=list(range(len(originals))),
            mode='markers+text',
            text=[f"{avg_similarity:.3f}"] * len(originals),
            name=f"{test} = {parameter}",
        ))

    fig.update_layout(
        title=title,
        xaxis_title=test,
        yaxis_title="Image Index",
        height=600,
        width=800,
        showlegend=True
    )
    fig.show()


# def draw_mnist(x: np.ndarray, y: np.ndarray, num_examples=10) -> None:
#     """
#     Visualizes a set of MNIST digit images with their corresponding labels.
#
#     This function displays a specified number of MNIST digit images in a row, with each image accompanied by its label.
#     It uses a grayscale color map for visualization.
#
#     :param x: A 2D numpy array where each row represents a flattened MNIST image of shape (784,).
#     :param y: A 1D numpy array of labels corresponding to the images in `x`.
#     :param num_examples: The number of images to display (default is 10).
#
#     :return: None. Displays the images using matplotlib.
#     """
#     plt.figure(figsize=(10, 1))
#     for i in range(num_examples):
#         plt.subplot(1, num_examples, i + 1)
#         plt.imshow(x[i].reshape(28, 28), cmap='gray')
#         plt.title(str(y[i]))
#         plt.axis('off')
#     plt.show()

