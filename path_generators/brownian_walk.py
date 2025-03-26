import torch

from typing import List, Tuple

from math import sqrt

delta = 30
dt = 10

def brownian_motion_path(start_pos: torch.Tensor, N: int) -> torch.Tensor:
    """
    Generates a 2D Brownian motion path starting from the given starting position.

    :param start_pos: A 2-element PyTorch tensor representing the starting x and y coordinates of the path.
    :param N: The number of points on the path to plot.
    :return: A 2D PyTorch tensor of shape (2, NUM_COORDINATES + 1), where each column represents the x and y coordinates
             of the path at each step, starting from start_pos.
    """
    # Generate a displacement vector with respect to the previous point for each point.
    steps = torch.normal(mean=0.0, std=delta * sqrt(dt), size=(2, N))
    # Generate a displacement vector with respect to the starting position.
    path = torch.cumsum(steps, dim=1)
    # Add the starting position to all the displacement vectors to generate the path.
    path = torch.cat((start_pos.unsqueeze(1), path + start_pos.unsqueeze(1)), dim=1)
    return path

def generate_path_with_points(path: torch.Tensor, num_points: int) -> List[Tuple[float, float]]:
    """
    Generates a specified number of equally spaced points along a given path.

    :param path: A 2D PyTorch tensor of shape (2, N) where each column represents the x and y coordinates of the path.
    :param num_points: The number of equally spaced points to generate along the path.
    :return: A list of tuples, where each tuple contains the x and y coordinates of an equally spaced point along the
    path.
    """
    # Calculate total path length
    total_path_length = torch.sum(torch.norm(path[:, 1:] - path[:, :-1], dim=0))

    # Starting point of the path
    points = [tuple(path[:, 0].tolist())]
    total_distance = 0.0

    # Calculate interval distance
    interval_distance = total_path_length / (num_points - 1)

    for i in range(1, path.size(1)):
        # Calculate the distance between two consecutive points on the path.
        segment_distance = torch.norm(path[:, i] - path[:, i - 1])
        # Continue this until the accumulated distance exceeds or matches the required interval distance.
        while total_distance + segment_distance >= interval_distance:
            # Calculate the ratio of the remaining distance to the length of the current segment.
            ratio = (interval_distance - total_distance) / segment_distance

            # Calculate the new point based on the ratio
            new_point = path[:, i - 1] + ratio * (path[:, i] - path[:, i - 1])

            # Append the newly calculated point to the list of points.
            points.append(tuple(new_point.tolist()))

            # Reduce the remaining segment distance by the interval distance.
            segment_distance -= interval_distance

            # Reset the accumulated distance after placing a new point.
            total_distance = 0.0

        total_distance += segment_distance
        if len(points) >= num_points:
            break

    return points

def plot_path(path: torch.Tensor, interval_points: List[Tuple[float, float]]) -> None:
    """
    Plots the 2D Brownian motion path along with the specified equally spaced points.

    :param path: A 2D PyTorch tensor of shape (2, N), where each column represents the x and y coordinates of the path.
    :param interval_points: A list of tuples, where each tuple represents the x and y coordinates of a point that is
                            equally spaced along the path.
    :return: None
    """
    # plt.plot(path[0].numpy(), path[1].numpy(), label='Brownian Path')
    # plt.scatter(*zip(*interval_points), color='purple', label='Interval Points')
    # plt.plot(path[0, 0].item(), path[1, 0].item(), 'go', label='Start Point')
    # plt.plot(path[0, -1].item(), path[1, -1].item(), 'ro', label='End Point')
    # plt.title('2D Brownian Motion with Equally Spaced Points')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

def generate_and_plot(path_length: int, N: int) -> List[Tuple[float, float]]:
    """
    Generates a 2D Brownian motion path starting from a random position and plots the path with the specified number of
    equally spaced points.

    :param path_length: The largest x and y coordinates of the path.
    :param N: The number of points to generate.

    :returns: A list of tuples, where each tuple represents the x and y coordinates of a point that is equally spaced
    along the path.
    """
    # Generate two random points (x, y) sampled from a uniform distribution.
    start_pos = torch.rand(2) * path_length
    # Get the brownian motion path and points.
    path = brownian_motion_path(start_pos, N)
    interval_points = generate_path_with_points(path, N)
    #plot_path(path, interval_points)

    return interval_points

# Example usage:
# generate_and_plot(5000, 10)
