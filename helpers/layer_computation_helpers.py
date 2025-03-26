import torch

def ReLU(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the ReLU activation function.

    :param x: The input tensor.
    :return: The output tensor after applying ReLU.
    """
    return torch.maximum(torch.tensor(0.0), x)


def norm_squared(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the squared norm (L2 norm) of a vector x.

    :param x: Input vector.
    :return: The squared norm of the vector x.
    """
    return torch.norm(x) ** 2


def incremental_update(W: torch.Tensor, input_vector: torch.Tensor, output: torch.Tensor, epsilon=1e-10) -> torch.Tensor:
    """
    Performs an incremental update to a weight matrix based on the given input and output tensors.

    This function updates the weight matrix `W` by adding a term that is the outer product of the output and input
    vectors, normalized by the squared norm of the input vector. A small epsilon value is added to prevent division by
    zero.

    :param W: A 2D tensor representing the weight matrix to be updated.
    :param input_vector: A 1D tensor representing the input vector.
    :param output: A 1D tensor representing the output vector.
    :param epsilon: A small float value added to the denominator to avoid division by zero (default is 1e-10).

    :return: A 2D tensor representing the updated weight matrix.
    """
    W += torch.outer(output, input_vector) / (norm_squared(input_vector) + epsilon)
    return W


def compute_p_from_s(s: torch.Tensor, Wps: torch.Tensor) -> torch.Tensor:
    """
    Given the sensory input s, compute the input to the hippocampus p.

    :param s: Sensory input pattern with shape (Ns,).
    :param Wps: Weights matrix with shape (Np, Ns).
    :return: Computed place cell pattern p with shape (Np,).
    """
    return ReLU(Wps @ s)


def compute_g_from_p(p: torch.Tensor, Wgp: torch.Tensor) -> torch.Tensor:
    """
    Using the hippocampus layer output p, compute the grid cell input g using the weights Wgp.

    :param p: Hippocampus layer output with shape (Np,).
    :param Wgp: Weights matrix with shape (Ng, Np).
    :return: Computed grid cell output g with shape (Ng,).
    """
    return Wgp @ p


def compute_p_from_g(g: torch.Tensor, Wpg: torch.Tensor) -> torch.Tensor:
    """
    Compute the hippocampus cell input p from the grid cell output g using the weights Wpg.

    :param g: Grid cell pattern with shape (Ng,).
    :param Wpg: Weights matrix with shape (Np, Ng).
    :return: Computed hippocampus layer input with shape (Np,).
    """
    return ReLU(Wpg @ g)


def compute_s_from_p(p: torch.Tensor, Wsp: torch.Tensor) -> torch.Tensor:
    """
    Compute the sensory layer input s using the hippocampus layer output p and the weights Wsp.

    :param p: Hippocampus cell output with shape (Np,).
    :param Wsp: Weights matrix with shape (Ns, Np).
    :return: Computed sensory layer input s with shape (Ns,).
    """
    return torch.tanh(Wsp @ p)
