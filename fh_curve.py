import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = "/Users/kd/Downloads/comp400/M4_results/datasets/hourly_data"  # Replace with your dataset file path
data = pd.read_csv(file_path, header=None, names=["Value"])

# Separate training and testing sets
training_values = data["Value"][:-48].values  # All but the last 48 values for training
test_values_actual = data["Value"][-48:].values  # Last 48 values for testing

# Set updated predictions
updated_predictions =[1629.3573, 1593.5984, 1570.5198, 1550.5537, 1533.2439, 1515.8403,
        1499.6282, 1484.6587, 1470.4729, 1457.0986, 1444.5344, 1432.7490,
        1421.6863, 1411.3196, 1401.6199, 1392.5538, 1384.0898, 1376.1971,
        1368.8462, 1362.0071, 1355.6510, 1349.7501, 1344.2772, 1339.2058,
        1334.5109, 1330.1680, 1326.1539, 1322.4469, 1319.0254, 1315.8700,
        1312.9613, 1310.2820, 1307.8151, 1305.5449, 1303.4565, 1301.5369,
        1299.7722, 1298.1512, 1296.6624, 1295.2954, 1294.0408, 1292.8896,
        1291.8335, 1290.8649, 1289.9767, 1289.1622, 1288.4159, 1287.7318]
def plot_test_predictions(training_values, test_values_actual, updated_predictions, output_file_prefix):
    # Non-zoomed plot
    plt.figure(figsize=(12, 6))

    # Number each entry instead of using dates
    training_indices = range(1, len(training_values) + 1)
    test_indices = range(len(training_values) + 1, len(training_values) + len(test_values_actual) + 1)

    # Plot the last 1/6 of the training data, excluding the testing section
    sixth_index = len(training_values) * 5 // 6
    plt.plot(training_indices[sixth_index:], training_values[sixth_index:], label="Training Data", color="blue")

    # Plot test data with their actual indices
    plt.plot(test_indices, test_values_actual, label="Test Data", color="orange", linestyle='--')

    # Plot updated predictions for the same test indices
    plt.plot(test_indices, updated_predictions, label="Updated Predictions", color="green", linestyle='dotted')

    # Add a shaded region for the test data
    plt.axvspan(test_indices[0], test_indices[-1], color="gray", alpha=0.2, label="Test Period")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Training data, test Data and Predictions for the hourly dataset: LSTM")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_file_prefix}_non_zoomed.png")
    plt.close()

    # Zoomed-in plot
    plt.figure(figsize=(12, 6))

    # Plot test data with their actual indices
    plt.plot(test_indices, test_values_actual, label="Test Data", color="orange", linestyle='--')

    # Plot updated predictions for the same test indices
    plt.plot(test_indices, updated_predictions, label="Updated Predictions", color="green", linestyle='dotted')

    # Add a shaded region for the test data
    plt.axvspan(test_indices[0], test_indices[-1], color="gray", alpha=0.2, label="Test Period")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Zoomed View of Test Data and Predictions for the hourly dataset: LSTM")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_file_prefix}_zoomed.png")
    plt.close()


plot_test_predictions(training_values, test_values_actual, updated_predictions, "hourly_lstm")
