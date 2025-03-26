import torch
import random

class LevyWalk:
    def __init__(self, alpha, beta, max_dist):
        self.alpha = alpha  # Stability parameter (0 < alpha <= 2)
        self.beta = beta  # Skewness parameter (-1 <= beta <= 1)
        self.max_dist = max_dist

        self.pos_x = torch.tensor(0.0)
        self.pos_y = torch.tensor(0.0)
        self.x_arr = [self.pos_x.item()]
        self.y_arr = [self.pos_y.item()]

    def levy_sample(self):
        """
        Generate a random step size using an approximation of the Lévy distribution.

        We use inverse transform sampling with PyTorch to generate stable distributions.
        """
        u = torch.rand(1) - 0.5  # Uniform random number in (-0.5, 0.5)
        v = torch.rand(1)  # Uniform random number in (0, 1)

        # Compute the Lévy-distributed step size
        step_size = u / torch.pow(v, 1 / self.alpha)

        # Ensure the step size is positive and apply scaling
        return torch.abs(step_size)

    def update_pos(self):
        step_size = torch.abs(torch.randn(1))  # Random step size
        step_size = min(self.max_dist, step_size.item())  # Limit step size

        angle = torch.tensor(random.uniform(0, 2 * torch.pi))  # Ensure angle is a tensor

        # Use torch.cos() and torch.sin() correctly with tensors
        self.pos_x += step_size * torch.cos(angle)
        self.pos_y += step_size * torch.sin(angle)

    def calculate_total_length(self):
        total_length = 0
        for i in range(1, len(self.x_arr)):
            dist = torch.sqrt(
                (self.x_arr[i] - self.x_arr[i - 1]) ** 2 +
                (self.y_arr[i] - self.y_arr[i - 1]) ** 2
            )
            total_length += dist.item()
        return total_length

    def get_equally_spaced_points(self, interval_distance, N):
        points = [(self.x_arr[0], self.y_arr[0])]
        total_distance = 0

        for i in range(1, len(self.x_arr)):
            dist = torch.sqrt(
                (self.x_arr[i] - self.x_arr[i - 1]) ** 2 +
                (self.y_arr[i] - self.y_arr[i - 1]) ** 2
            )
            total_distance += dist.item()

            if total_distance >= interval_distance:
                points.append((self.x_arr[i], self.y_arr[i]))
                total_distance = 0

            if len(points) >= N:
                break

        return points

    def generate_path(self, steps, N):
        for _ in range(steps):
            self.update_pos()
            if self.calculate_total_length() >= self.max_dist:
                break

        return self.get_equally_spaced_points(steps // N, N)
    # def show_path(self, points):
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(self.x_arr, self.y_arr, linestyle='-', color='blue', label='Lévy Walk Path')
    #     plt.scatter(*zip(*points), color='red', s=50, label='Equally Spaced Points')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Lévy Walk in 2D with Equally Spaced Points')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

# Example usage:
# levy_walk = LevyWalk(alpha=1.5, beta=0, max_dist=10000)
# points = levy_walk.generate_path(50000, 100)
# levy_walk.show_path(points)
