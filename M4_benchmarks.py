import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


# Detrending function
def detrend(insample_data):
    x = torch.arange(len(insample_data)).float()
    a, b = torch.polyfit(x, insample_data, 1)
    return a.item(), b.item()


# Deseasonalizing function
def deseasonalize(original_ts, ppy):
    if seasonality_test(original_ts, ppy):
        ma_ts = moving_averages(original_ts, ppy)
        le_ts = (original_ts * 100) / ma_ts
        le_ts = torch.cat([le_ts, torch.full((ppy - (len(le_ts) % ppy),), float('nan'))])
        le_ts = le_ts.view(-1, ppy)
        si = torch.nanmean(le_ts, dim=0)
        norm = torch.sum(si) / (ppy * 100)
        si = si / norm
    else:
        si = torch.full((ppy,), 100)
    return si


# Moving averages function
def moving_averages(ts_init, window):
    if len(ts_init) % 2 == 0:
        ts_ma = pd.Series(ts_init).rolling(window=window, center=True).mean()
        ts_ma = pd.Series(ts_ma).rolling(window=2, center=True).mean()
        ts_ma = torch.roll(ts_ma, -1)
    else:
        ts_ma = pd.Series(ts_init).rolling(window=window, center=True).mean()
    return torch.tensor(ts_ma.values)


# Seasonality test function
def seasonality_test(original_ts, ppy):
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)
    limit = 1.645 * ((1 + 2 * s) / len(original_ts)).sqrt()
    return abs(acf(original_ts, ppy)) > limit


def moving_averages(original_ts, ppy):
    # Ensure the input is a tensor for torch operations
    if isinstance(original_ts, pd.Series):
        original_ts = torch.tensor(original_ts.values, dtype=torch.float32)

    # Create a tensor for the moving averages
    ts_ma = original_ts.clone()

    # Apply torch.roll (this is just an example of how you might use it)
    ts_ma = torch.roll(ts_ma, shifts=-1)

    # Continue with other moving average calculations as needed

    return ts_ma

# Autocorrelation function
def acf(data, k):
    m = torch.mean(data)
    s1 = sum((data[i] - m) * (data[i - k] - m) for i in range(k, len(data)))
    s2 = sum((data[i] - m) ** 2 for i in range(len(data)))
    return s1 / s2


# Splitting data into train/test sets
def split_into_train_test(data, in_num, fh):
    train, test = data[:-fh], data[-(fh + in_num):]
    x_train, y_train = train[:-1], train[in_num:]
    x_test, y_test = train[-in_num:], test[in_num:]

    x_train = x_train.unsqueeze(1).float()
    x_test = x_test.unsqueeze(1).float()

    return x_train, y_train, x_test, y_test


# RNN benchmark
class RNNModel(nn.Module):
    def __init__(self, input_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=6, batch_first=True)
        self.fc = nn.Linear(6, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def rnn_bench(x_train, y_train, x_test, fh, input_size):
    model = RNNModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    x_train = x_train.view(-1, input_size, 1)
    x_test = x_test.view(-1, input_size, 1)

    # Train the RNN
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train.float())
        loss.backward()
        optimizer.step()

    # Forecasting
    y_hat_test = []
    last_prediction = model(x_test).item()
    for i in range(fh):
        y_hat_test.append(last_prediction)
        x_test = torch.roll(x_test, shifts=-1, dims=1)
        x_test[:, -1, :] = last_prediction
        last_prediction = model(x_test).item()

    return torch.tensor(y_hat_test)


# MLP benchmark
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(1, 6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def mlp_bench(x_train, y_train, x_test, fh):
    model = MLPModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the MLP
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x_train.float())
        loss = criterion(outputs, y_train.float())
        loss.backward()
        optimizer.step()

    # Forecasting
    y_hat_test = []
    last_prediction = model(x_test).item()
    for i in range(fh):
        y_hat_test.append(last_prediction)
        x_test = torch.roll(x_test, shifts=-1, dims=1)
        x_test[:, -1, :] = last_prediction
        last_prediction = model(x_test).item()

    return torch.tensor(y_hat_test)


# sMAPE calculation function

def smape(a, b):
    print(f"Shape of a: {a.shape}, Shape of b: {b.shape}")

    # Flatten the tensors
    a = a.view(-1)
    b = b.view(-1)

    # Calculate sMAPE
    return torch.mean(
        2.0 * torch.abs(a - b) / (torch.abs(a) + torch.abs(b) + 1e-8))  # Small constant to avoid division by zero


# MASE calculation function
def mase(insample, y_test, y_hat_test, freq):
    # Ensure insample is a tensor
    if not isinstance(insample, torch.Tensor):
        insample = torch.tensor(insample, dtype=torch.float32)

    # Generate the naive forecast using past values with frequency steps
    y_hat_naive = insample[:-freq]  # Slices from the start up to `-freq`
    y_true_naive = insample[freq:]  # Slices from `freq` to the end

    # Calculate the denominator term for MASE
    masep = torch.mean(torch.abs(y_true_naive - y_hat_naive))

    # Calculate the MASE metric
    return torch.mean(torch.abs(y_test - y_hat_test)) / masep


def main():
    fh = 6  # forecasting horizon
    freq = 1  # data frequency
    in_size = 3  # number of points used as input for each forecast

    err_MLP_sMAPE = []
    err_MLP_MASE = []
    err_RNN_sMAPE = []
    err_RNN_MASE = []

    # ===== In this example we produce forecasts for 100 randomly generated timeseries =====
    data_all = torch.randint(0, 100, (100, 20)).float()
    for i in range(0, 100):
        for j in range(0, 20):
            data_all[i, j] = j * 10 + data_all[i, j]

    counter = 0
    # ===== Main loop which goes through all timeseries =====
    for j in range(len(data_all)):
        ts = data_all[j, :]

        # remove seasonality
        seasonality_in = deseasonalize(ts, freq)

        for i in range(len(ts)):
            ts[i] = ts[i] * 100 / seasonality_in[i % freq]

        # detrending
        a, b = detrend(ts)
        for i in range(len(ts)):
            ts[i] = ts[i] - ((a * i) + b)

        x_train, y_train, x_test, y_test = split_into_train_test(ts, in_size, fh)

        # RNN benchmark - Produce forecasts
        y_hat_test_RNN = rnn_bench(x_train, y_train, x_test, fh, in_size)

        # MLP benchmark - Produce forecasts
        y_hat_test_MLP = mlp_bench(x_train, y_train, x_test, fh)
        for i in range(29):
            y_hat_test_MLP = torch.vstack([y_hat_test_MLP, mlp_bench(x_train, y_train, x_test, fh)])
        y_hat_test_MLP = torch.median(y_hat_test_MLP, dim=0).values

        # add trend
        for i in range(len(ts)):
            ts[i] = ts[i] + ((a * i) + b)

        for i in range(fh):
            y_hat_test_MLP[i] = y_hat_test_MLP[i] + ((a * (len(ts) + i + 1)) + b)
            y_hat_test_RNN[i] = y_hat_test_RNN[i] + ((a * (len(ts) + i + 1)) + b)

        # add seasonality
        for i in range(len(ts)):
            ts[i] = ts[i] * seasonality_in[i % freq] / 100

        for i in range(len(ts), len(ts) + fh):
            y_hat_test_MLP[i - len(ts)] = y_hat_test_MLP[i - len(ts)] * seasonality_in[i % freq] / 100
            y_hat_test_RNN[i - len(ts)] = y_hat_test_RNN[i - len(ts)] * seasonality_in[i % freq] / 100

        # check if negative or extreme
        for i in range(len(y_hat_test_MLP)):
            if y_hat_test_MLP[i] < 0:
                y_hat_test_MLP[i] = 0
            if y_hat_test_RNN[i] < 0:
                y_hat_test_RNN[i] = 0
            if y_hat_test_MLP[i] > (1000 * max(ts)):
                y_hat_test_MLP[i] = max(ts)
            if y_hat_test_RNN[i] > (1000 * max(ts)):
                y_hat_test_RNN[i] = max(ts)

        # Calculate errors
        err_MLP_sMAPE.append(smape(y_test, y_hat_test_MLP).item())
        err_RNN_sMAPE.append(smape(y_test, y_hat_test_RNN).item())
        err_MLP_MASE.append(mase(ts[:-fh], y_test, y_hat_test_MLP, freq).item())
        err_RNN_MASE.append(mase(ts[:-fh], y_test, y_hat_test_RNN, freq).item())

        counter += 1
        print(f"-------------TS ID: {counter} -------------")

    print("\n\n---------FINAL RESULTS---------")
    print("=============sMAPE=============\n")
    print("#### MLP ####\n", torch.mean(torch.tensor(err_MLP_sMAPE)), "\n")
    print("#### RNN ####\n", torch.mean(torch.tensor(err_RNN_sMAPE)), "\n")
    print("==============MASE=============")
    print("#### MLP ####\n", torch.mean(torch.tensor(err_MLP_MASE)), "\n")
    print("#### RNN ####\n", torch.mean(torch.tensor(err_RNN_MASE)), "\n")


if __name__ == "__main__":
    main()
