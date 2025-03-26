import torch
import random

from typing import List, Tuple

def generate_points(path_length: int, N: int) -> List[Tuple[float, float]]:
    """
    Generates a specified number of equally spaced points along a random path (in a line) defined by a series of random
    coordinates.

    :param path_length: The largest x and y coordinates of the path.
    :param N: The number of points to generate.

    :returns: A list of tuples, where each tuple represents the x and y coordinates of a point that is equally spaced
    along the path.
    """
    # Generate random coordinates for x and y with each coordinate being a random integer between 1 and path_length.
    rand_x = [random.randint(1, path_length) for _ in range(N)]
    rand_y = [random.randint(1, path_length) for _ in range(N)]

    # Convert to PyTorch tensors and sort them in ascending order.
    x = torch.tensor(rand_x).sort().values
    y = torch.tensor(rand_y).sort().values

    # This will be the starting point of the path.
    points = [(x[0].item(), y[0].item())]
    total_distance = 0.0

    # Calculate the total length of the path by summing the Euclidean distances between consecutive points.
    path_length_total = torch.sum(torch.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))

    # Calculate the interval distance, which is the distance that should separate each of the num_points points.
    interval_distance = path_length_total / (N - 1)

    # Iterate through each segment formed by the consecutive points in the sorted x and y arrays.
    for i in range(1, len(x)):
        # Calculate the distance of the current segment between points (x[i-1], y[i-1]) and (x[i], y[i]).
        segment_distance = torch.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)

        # Place new points along the segment at intervals defined by interval_distance.
        # Continue this until the accumulated distance exceeds or matches the required interval distance.
        while total_distance + segment_distance >= interval_distance:
            # Calculate the ratio of the remaining distance to the length of the current segment.
            ratio = (interval_distance - total_distance) / segment_distance

            # Interpolate the x and y coordinates for the new point along the segment.
            new_x = x[i - 1] + ratio * (x[i] - x[i - 1])
            new_y = y[i - 1] + ratio * (y[i] - y[i - 1])

            # Append the newly calculated point to the list of points.
            points.append((new_x.item(), new_y.item()))

            # Reset the accumulated distance since the last point to 0 after placing a new point.
            total_distance = 0.0

            # Reduce the remaining segment distance by the interval distance, and continue placing points.
            segment_distance -= interval_distance

        # Accumulate the remaining distance in the segment after placing the last point.
        total_distance += segment_distance

        # If the required number of points has been generated, exit the loop early.
        if len(points) >= N:
            break

    # Return the list of generated points.
    return points

# def plot_interval_points(interval_points: List[Tuple[float, float]]) -> None:
#     """
#     Plots the given interval points along the path and connects them with a line.
#
#     :param interval_points: A list of tuples, where each tuple represents the x and y coordinates of a point that is
#                             equally spaced along the path.
#     :return: None
#     """
#     plt.title("Equally Spaced Points")
#     plt.xlabel("x")
#     plt.ylabel("y")
#
#     # Unzip the interval_points into separate x and y lists
#     x_values, y_values = zip(*interval_points)
#
#     # Plot the points as red dots
#     plt.scatter(x_values, y_values, color='red')
#
#     # Draw a line connecting the points
#     plt.plot(x_values, y_values, color='blue')
#
#     # Display the plot
#     plt.show()

# Example Usage
# path_length = 30
# Generate and plot the points
# interval_points = generate_points(path_length, 10)
# plot_interval_points(interval_points)
