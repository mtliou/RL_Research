import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import warnings
import random
from path_generators.regular_line import  generate_points
from path_generators.brownian_walk import generate_and_plot
from path_generators.levy_flight import LevyWalk

warnings.filterwarnings('ignore')

class GridConfig:
    def __init__(self, N, lambda1, lambda2, lambda3, path_type, movements, Nh, corruption_level, length, grid_type):
        self.N = N  # Number of samples to show
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.path_type = path_type
        self.movements = movements or []
        self.Nh = Nh
        self.corruption_level = corruption_level
        self.length = length
        self.grid_type = grid_type

    def run(self):
        """Generate path based on the specified type."""
        if self.path_type.lower() == 'levy':
            walker = LevyWalk(alpha=1.5, beta=0, max_dist=10000)
            movements = walker.generate_path(steps=50000, N=100)
            self.movements.append(movements)

        elif self.path_type.lower() == 'brownian':
            movements = generate_and_plot()
            self.movements.append(movements)

        elif self.path_type.lower() == 'line':
            rand_x = [random.randint(1, 600) for _ in range(600)]
            rand_y = [random.randint(1, 600) for _ in range(600)]
            movements = generate_points(rand_x, rand_y)
            self.movements.append(movements)

        else:
            raise ValueError(f"Invalid path type: {self.path_type}")

        print(f"Generated {len(self.movements)} movement paths.")

# Initialize grid configuration
grid = GridConfig(50, 0.1, 0.2, 0.3, 'Levy', [], 1000, 'medium', 10000, 'hex')
grid.run()

# Load CIFAR-100 dataset using torchvision
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

def show_samples(grid: GridConfig, data_loader):
    """Display a grid of sample images from the dataset using PIL."""
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))

    # Create a grid of images using make_grid from torchvision
    image_grid = make_grid(images, nrow=5)  # 5 images per row

    # Convert the grid to a PIL image
    pil_image = transforms.ToPILImage()(image_grid)

    # Display the image using PIL's show method
    pil_image.show()

# Show samples from the train_loader
show_samples(grid, train_loader)
