import matplotlib.pyplot as plt
import pandas as pd

# test_dates_actual, test_values_actual, training_dates, training_values, updated_predictions

output_file = "daily_lstm"
# Load the training dataset from a CSV file
data = pd.read_csv("/Users/kd/Downloads/comp400/M4_results/datasets/daily_data")
training_dates = pd.to_datetime(data["Date"])  # Convert dates to datetime
training_values = data["Temp"].values  # Extract temperature values

# Test data: Use the last 14 values from training data as the test set
test_values_actual = training_values[-14:]
test_dates_actual = training_dates[-14:]  # Corresponding dates for the test values

# Predictions: list of predictions, the first values of each arrays taken
updated_predictions = [ 8.9938,  9.2325,  9.4850,  9.7183,  9.9260, 10.1095, 10.2719, 10.4161,
        10.5446, 10.6598, 10.7633, 10.8568, 10.9411, 11.0178]



def plot_test_predictions(training_dates, training_values, test_dates_actual, test_values_actual, updated_predictions):
    # Non-zoomed plot
    plt.figure(figsize=(12, 6))

    # Plot the last 1/6 of the training data, excluding the testing section
    sixth_index = len(training_dates) * 5 // 6
    plt.plot(training_dates[sixth_index:-14], training_values[sixth_index:-14], label="Training Data", color="blue")

    # Plot test data with their actual dates
    plt.plot(test_dates_actual, test_values_actual, label="Test Data", color="orange", linestyle='--')

    # Plot updated predictions for the same test dates
    plt.plot(test_dates_actual, updated_predictions, label="Predictions", color="green", linestyle='dotted')

    # Add a shaded region for the test data
    plt.axvspan(test_dates_actual.iloc[0], test_dates_actual.iloc[-1], color="gray", alpha=0.2, label="Test Period")

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Training data, test Data and Predictions for the Daily dataset: LSTM")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_file}_non_zoomed.png")
    plt.close()
    plt.show()

    # Zoomed-in plot
    plt.figure(figsize=(12, 6))

    # Plot test data with their actual dates
    plt.plot(test_dates_actual, test_values_actual, label="Test Data", color="orange", linestyle='--')

    # Plot updated predictions for the same test dates
    plt.plot(test_dates_actual, updated_predictions, label="Predictions", color="green", linestyle='dotted')

    # Add a shaded region for the test data
    plt.axvspan(test_dates_actual.iloc[0], test_dates_actual.iloc[-1], color="gray", alpha=0.2, label="Test Period")

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Zoomed View of Test Data and Predictions for the Daily dataset: LSTM")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_file}_zoomed.png")
    plt.close()

# Call the function
plot_test_predictions(training_dates, training_values, test_dates_actual, test_values_actual, updated_predictions)
