import torch
import torch.nn as nn
from M4_benchmarks import  split_into_train_test
from OneDg.OD_grid_utils import get_sensory_inputs, make_sequence

#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/hourly_data'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/daily_data'
file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/weekly_data'
#file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/monthly_data'
# file = '/home/mcb/users/kduran1/temporal/TemporalNeuroAI-revised_torch_model/dataset/yearly_data'

input = make_sequence(file)
input = [float(x) for x in input]  # Convert strings to floats
# Convert to a PyTorch tensor
sensory_input = torch.tensor(input, dtype=torch.float32)

def smape(actual, predicted):
    """
    Compute the symmetric mean absolute percentage error (sMAPE).
    """
    # Ensure inputs are tensors
    if isinstance(actual, list):
        actual = torch.cat(actual) if isinstance(actual[0], torch.Tensor) else torch.tensor(actual, dtype=torch.float32)
    if isinstance(predicted, list):
        predicted = torch.cat(predicted) if isinstance(predicted[0], torch.Tensor) else torch.tensor(predicted, dtype=torch.float32)
    actual = actual.view(-1)
    predicted = predicted.view(-1)

    # Avoid division by zero
    epsilon = 1e-8
    numerator = 2.0 * torch.abs(actual - predicted)
    denominator = torch.abs(actual) + torch.abs(predicted) + epsilon
    return torch.mean(numerator / denominator).item()


# Function to calculate MASE
def mase(insample, y_test, y_hat_test, freq):
    """
    Compute the mean absolute scaled error (MASE).
    """
    # Ensure inputs are tensors
    if not isinstance(insample, torch.Tensor):
        insample = torch.tensor(insample, dtype=torch.float32)

    if isinstance(y_test, list):
        y_test = torch.stack(y_test) if isinstance(y_test[0], torch.Tensor) else torch.tensor(y_test, dtype=torch.float32)
    if isinstance(y_hat_test, list):
        y_hat_test = torch.stack(y_hat_test) if isinstance(y_hat_test[0], torch.Tensor) else torch.tensor(y_hat_test, dtype=torch.float32)

    y_test = y_test.view(-1)
    y_hat_test = y_hat_test.view(-1)

    # Generate naive forecast
    y_hat_naive = insample[:-freq]
    y_true_naive = insample[freq:]

    masep = torch.mean(torch.abs(y_true_naive - y_hat_naive))
    return torch.mean(torch.abs(y_test - y_hat_test)) / masep

#hourly
# fh=48
# in_num = 700
#daily
# fh=14
# in_num = 93
# #weekly
fh=13
in_num = 80
# #monthly
# fh=18
# in_num = 42
# #yearly
# fh= 6
# in_num = 13

# Split into train and test sets (before normalization)
x_train, y_train, x_test, y_test = split_into_train_test(sensory_input, in_num, fh)

# Compute mean and std using ONLY the training set
train_mean = x_train.mean()
train_std = x_train.std()

# Normalize train and test sets using training statistics
x_train = (x_train - train_mean) / train_std
x_test = (x_test - train_mean) / train_std  # Use train mean/std to avoid leakage

# Normalize y_train and y_test if needed
y_train = (y_train - train_mean) / train_std
y_test = (y_test - train_mean) / train_std

seq_len = fh  # Number of time steps per sequence
x_train_sequences = x_train.unfold(0, seq_len, 1).permute(0, 2, 1)

# Add noise to the training data (data augmentation)
noisy_input = x_train_sequences + 0.1 * torch.randn_like(x_train_sequences)

batch_size = 4
num_batches = len(x_train_sequences) // batch_size
x_train_batches = torch.split(x_train_sequences, batch_size)
noisy_batches = torch.split(noisy_input, batch_size)

# Define the basic GRU-based model
class GRURecallModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRURecallModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.gru(x)
        reconstructed = self.fc(out)
        return reconstructed

input_size = 1
hidden_size = 512
num_layers = 2
learning_rate = 0.001
num_epochs = 30

model = GRURecallModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
#tis loss is better for forecast than mse
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx in range(num_batches):
        batch_noisy = noisy_batches[batch_idx]
        batch_clean = x_train_batches[batch_idx]

        # Forward pass
        reconstructed = model(batch_noisy)
        loss = criterion(reconstructed, batch_clean)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / num_batches:.4f}")

# Prepare test sequences
x_test_sequences = x_test.unfold(0, seq_len, 1).permute(0, 2, 1)

# Iterative Rolling Forecast, like the one the M4 used for it's RNN baseline
rolling_predictions = []
model.eval()
with torch.no_grad():
    current_input = x_test_sequences[0:1].clone()  # Start with the first test sequence (batch of 1)

    for i in range(fh):  # Forecast `fh` steps iteratively
        # Predict the next value using the last time step of the GRU output
        prediction = model(current_input)[:, -1, 0]  # Output the last step's value
        rolling_predictions.append(prediction.item())  # Store the scalar prediction

        # Update the input sequence for the next step
        new_input = torch.roll(current_input, shifts=-1, dims=1).clone()  # Shift left
        new_input[:, -1, 0] = prediction  # Replace the last value with the new prediction
        current_input = new_input  # Update current_input for the next iteration

# Convert predictions to tensor and denormalize using TRAINING mean and std
rolling_predictions = torch.tensor(rolling_predictions) * train_std + train_mean

# Denormalize
y_test = y_test * train_std + train_mean
print(y_test)

# Compute SMAPE
smape_value = smape(y_test, rolling_predictions)
print(f"Rolling Predictions (iterative): {rolling_predictions}")
print(f"SMAPE: {smape_value}")

# Compute MASE
mase_value = mase(insample=x_train, y_test=y_test, y_hat_test=rolling_predictions, freq=fh)
print(f"MASE: {mase_value}")
