import torch
from typing import List
#from model.model import memorize, Memory
# Set print options to display the entire tensor
torch.set_printoptions(threshold=torch.inf)

##initialization works well
##move works well
class OneDGrid:
    """
    A collection of the grid modules in a 1D array.

    ============================================= Class Attributes =============================================
    type: The type of grid cell being used (square or hexagon).
    lambdas: A list of sizes of integers determining the 1D grid size.
    N_g: The total size of the grid module, which is the sum of all lambdas.
    g: The grid tensor containing all the grid modules.
    """

    type: str
    lambdas: List[int]
    N_g: int
    g: torch.Tensor

    def __init__(self, lambdas: List[int]) -> None:
        """
        Initialize a OneDGrid object.

        :param lambdas: A list of sizes for each segment of the 1D grid.
        """
        self.type = "1D"
        self.lambdas = lambdas
        self.N_g = sum(self.lambdas)  # Sum of all lambdas
        self.g = torch.zeros(self.N_g)  # Initialize the grid with zeros
        self.N_modules = len(self.lambdas)  # The number of grid modules

        # Initialize the first active position in each segment
        index = 0
        for size in self.lambdas:
            self.g[index] = 1  # Set the starting point in each segment
            index += size

    def move(self, steps: int = 1) -> None:
        """
        Move the active '1' in every segment according to its size, with wrap-around.

        :param steps: Number of steps to move each active element forward.
        """
        new_grid = []
        start = 0

        for size in self.lambdas:
            segment = self.g[start:start + size]  # Extract the segment

            # What will be the new index 0 for this segment
            new_zero = - steps % size

            # Rotate all the values around
            rotated_segment = torch.cat([segment[new_zero:], segment[:new_zero]])
            # Add the new segment to the new grid
            new_grid.append(rotated_segment)

            start += size

        # Concatenate all segments into a single tensor
        self.g = torch.cat(new_grid)

    def move_multiple_times(self, num_moves: int) -> None:
        """
        Perform multiple moves sequentially by calling the move function repeatedly.

        :param num_moves: Number of times to call the move function.
        """
        for i in range(num_moves):
            print(f"\nGrid State after moving {i + 1} time(s):")
            self.move(steps=1)  # Always move by 1 step per call

    def __str__(self):
        return '1D'


#################
##TEST FOR MOVE##
#################
    def move2(self, start: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Move the active positions in the grid based on the provided start tensor,
        moving active positions forward by the specified number of steps within
        their respective segments.

        :param start: A tensor representing the initial active positions.
        :param steps: Number of steps to move each active position forward.
        """
        if start.shape != self.g.shape:
            raise ValueError("The 'start' tensor must have the same shape as the grid 'g'.")

        new_grid = []
        current_start = 0  # Tracks the start of each segment in the grid

        for size in self.lambdas:
            segment = self.g[current_start:current_start + size]  # Extract the grid segment
            start_segment = start[current_start:current_start + size]  # Extract the corresponding start segment

            # Determine the active position in the start segment
            active_indices = (start_segment == 1).nonzero(as_tuple=True)[0]
            if len(active_indices) != 1:
                raise ValueError(f"Each segment must have exactly one active position. Found: {len(active_indices)}.")

            active_index = active_indices.item()

            # Determine the new active position after moving by 'steps'
            new_active_index = (active_index + steps) % size

            # Create the new segment with the active position moved
            rotated_segment = torch.zeros_like(segment)
            rotated_segment[new_active_index] = 1

            # Add the updated segment to the new grid
            new_grid.append(rotated_segment)

            current_start += size  # Move to the next segment

        # Concatenate all updated segments into the new grid
        g = torch.cat(new_grid)
        self.g = g
        return g

    def move2_return(self,start: torch.Tensor, steps: int = 1) -> List[torch.Tensor]:
        if start.shape != self.g.shape:
            raise ValueError("The 'start' tensor must have the same shape as the grid 'g'.")
        grids = []
        for i in range(steps):
            g = self.move2(start,i)
            grids.append(g)
        return grids



if __name__ == "__main__":
    # Example lambdas list
    w = 6  # Parameter for sensory input
    lambdas = [2, 3] #, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              # 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    # Initialize the grid
    grid = OneDGrid(lambdas)
    grid.move2(torch.tensor([1., 0., 1., 0., 0.]), 1)
    print(grid.g)
