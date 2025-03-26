import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Define test values as a PyTorch tensor
test_values = torch.tensor([
    [1559., 1559., 1559., 1559., 1559., 1558.],
    [1559., 1559., 1559., 1559., 1559., 1559.],
    [1559., 1559., 1559., 1559., 1559., 1559.],
    [1559., 1559., 1559., 1559., 1559., 1559.],
    [1098., 1559., 1559., 1559., 1559., 1559.],
    [1112., 1098., 1559., 1559., 1559., 1559.],
    [1109., 1112., 1098., 1559., 1559., 1559.],
    [1111., 1109., 1112., 1098., 1559., 1559.],
    [1103., 1111., 1109., 1112., 1098., 1559.],
    [1113., 1103., 1111., 1109., 1112., 1098.],
    [1119., 1113., 1103., 1111., 1109., 1112.],
    [1125., 1119., 1113., 1103., 1111., 1109.],
    [1122., 1125., 1119., 1113., 1103., 1111.],
    [1107., 1122., 1125., 1119., 1113., 1103.],
    [1110., 1107., 1122., 1125., 1119., 1113.],
    [1104., 1110., 1107., 1122., 1125., 1119.],
    [1108., 1104., 1110., 1107., 1122., 1125.],
    [1105., 1108., 1104., 1110., 1107., 1122.]
])

# Define recall values as a list of PyTorch tensors
recall_values = [
    torch.tensor([1558.9220, 1558.9220, 1558.7310, 1558.7306, 1558.7306, 1558.7306]),
    torch.tensor([1558.8065, 1558.8058, 1558.7614, 1558.7631, 1558.7614, 1558.7888]),
    torch.tensor([1558.7404, 1558.7639, 1558.7649, 1558.7333, 1558.7170, 1558.7631]),
    torch.tensor([1558.6956, 1558.7357, 1558.7291, 1558.7122, 1558.7198, 1558.6667]),
    torch.tensor([1558.6785, 1558.6759, 1558.6770, 1558.6940, 1558.6952, 1558.6947]),
    torch.tensor([1558.5713, 1558.6694, 1558.6547, 1558.6727, 1558.6708, 1558.5968]),
    torch.tensor([1558.5011, 1558.5934, 1558.6433, 1558.6581, 1558.6256, 1558.5609]),
    torch.tensor([1558.5800, 1558.5696, 1558.5559, 1558.5917, 1558.6016, 1558.5479]),
    torch.tensor([1558.5575, 1558.5415, 1558.5417, 1558.5441, 1558.5574, 1558.5577]),
    torch.tensor([1558.4668, 1558.5269, 1558.5322, 1558.5320, 1558.4692, 1558.5270]),
    torch.tensor([1558.3898, 1558.4778, 1558.4910, 1558.5034, 1558.5037, 1558.3958]),
    torch.tensor([1558.4647, 1558.4297, 1558.4324, 1558.4301, 1558.4644, 1558.4541]),
    torch.tensor([1558.3990, 1558.3989, 1558.4119, 1558.4135, 1558.4198, 1558.4199]),
    torch.tensor([1558.3615, 1558.3617, 1558.3621, 1558.3854, 1558.3898, 1558.3898]),
    torch.tensor([1558.3300, 1558.3296, 1558.3418, 1558.3307, 1558.3484, 1558.3539]),
    torch.tensor([1558.2812, 1558.3184, 1558.2798, 1558.3162, 1558.3046, 1558.3062]),
    torch.tensor([1558.2510, 1558.2682, 1558.2787, 1558.2791, 1558.2684, 1558.2504]),
    torch.tensor([1558.2391, 1558.2400, 1558.2400, 1558.2400, 1558.2346, 1558.2346])
]

# Stack recall values into a tensor
recall_tensor = torch.stack(recall_values)

# Calculate the total sum of all recall tensors
total_recall_sum = torch.sum(recall_tensor)

# Compute absolute differences
heatmap_data = torch.abs(test_values[:len(recall_tensor)] - recall_tensor)

# Convert to NumPy for visualization
heatmap_data_np = heatmap_data.numpy()

# Print the total sum
print("Total sum of recall tensors:", total_recall_sum.item())

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data_np, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Resemblance Heatmap (Absolute Differences)")
plt.xlabel("Index")
plt.ylabel("Test/Recall Pair")
plt.savefig("heatmap_plot_p4_month.png")  # Save the plot as an image
plt.show()

