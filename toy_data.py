import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

num_points = 3650
num_periods = 5
sin_range = 100
nonneg_vals = True

# Generate equally spaced points
x = np.linspace(0, num_periods * (2 * np.pi), num_points)

# Generate the scaled sinusoidal values from [-sin_range, sin_range]
if nonneg_vals:
    y = np.sin(x) * sin_range + sin_range
else:
    y = np.sin(x) * sin_range

# Store the data to the dataset subdirectory
filename = 'toy_data'
subdir = 'dataset'
filepath = os.path.join(subdir, filename)

data = pd.DataFrame(y, columns=['sin_value'])
data.to_csv(filepath, index=False, header=False, sep='\n')
print(f'Toy dataset saved to dataset subdirectory with filename: {filename}')

# Plot dataset for visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue')
plt.title('Sinusoidal Pattern')
plt.xlabel('x (radians)')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()
plt.savefig(f'{filename}_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'Toy dataset plot saved with filename: {filename}_plot.png')