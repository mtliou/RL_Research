import torch

file = "/home/mcb/users/kduran1/temporal_torch/te"

def get_sensory_inputs(filename: str, w: int):
    """
    Args:
        filename: path to the csv file
        w: hyperparameter deciding the memory trail length.

    Returns:
        Sensory inputs stored in a tensor, where each input is made of the current value `t`
        as the first element, followed by `t-1, t-2, ..., t-w` memories, padded with zeros if necessary.
    """
    # Assuming make_sequence correctly processes the filename and returns a sequence
    sequence = make_sequence(filename)
    s = []  # Initialize an empty list to store the sensory inputs
    sequence = torch.tensor([float(val) for val in sequence], dtype=torch.float32)

    for t in range(len(sequence)):
        # Collect the current value `t`
        current = sequence[t:t + 1]
        # Collect the past w elements including `t` but in reverse order
        past_memories = sequence[max(0, t - (w - 1)):t].flip(0)  # Extract and reverse the past values
        # Pad the past memories with zeros if there are fewer than w-1 memories
        if len(past_memories) < w - 1:
            padding = torch.zeros(w - 1 - len(past_memories), dtype=torch.float32)
            past_memories = torch.cat((past_memories, padding))  # Pad zeros at the end of the reversed memories
        # Combine `t` as the first element, followed by the reversed past memories
        element = torch.cat((current, past_memories))
        # Append the trail to the list `s`
        s.append(element)

    s = torch.stack(s)
    return s


def make_sequence(file):
    """

    Args:
        file: The csv file from where to extract the data from.

    Returns:
        A list of strings containing all the data, in order.

    """
    combined_sequence = []
    # Open the file and read each line
    with open(file, 'r') as file:
        for index, line in enumerate(file):
            # Skip the first line
            if index == 0:
                continue
            # Split the line by comma and get the last element, stripping any quotes and spaces
            last_value = line.split(',')[-1].strip().replace('"', '')
            combined_sequence.append(last_value)
    return combined_sequence


def get_stats():
    """
    Returns: Standard deviation and mean as a tensor.
    """
    data = make_sequence(file)
    # Step 1: Convert strings to floats
    float_list = [float(num) for num in data]
    # Step 2: Convert list of floats to tensor
    tensor = torch.tensor(float_list)
    std_dev = torch.std(tensor)
    mean = torch.mean(tensor)

    return std_dev, mean

def smape(a, b):
    """
    Calculates sMAPE (Symmetric Mean Absolute Percentage Error)
    :param a: actual values
    :param b: predicted values
    :return: sMAPE score
    """
    a = a.view(-1)
    b = b.view(-1)
    return torch.mean(2.0 * torch.abs(a - b) / (torch.abs(a) + torch.abs(b)))


def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE (Mean Absolute Scaled Error)
    :param insample: in-sample data
    :param y_test: out-of-sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return: MAsE score
    """
    y_hat_naive = [insample[i - freq] for i in range(freq, len(insample))]
    masep = torch.mean(torch.abs(insample[freq:] - torch.tensor(y_hat_naive)))
    return torch.mean(torch.abs(y_test - y_hat_test)) / masep

