import numpy as np
import matplotlib.pyplot as plt
from dataset_helpers import load_cifar10, randomize_arrays_by_class
from plot_graphs_helpers import plot_results


def reverse_transform_cifar_representation(transformed_flattened: np.ndarray) -> np.ndarray:
    """
    Reverses the transformation applied to CIFAR images, attempting to recover the original RGB image.

    :param transformed_flattened: A flattened numpy array representing the transformed CIFAR image.
    :return: The recovered RGB image in its original 32x32x3 shape.
    """
    # Reshape the input into the 32x32x4 format (Illumination, RG_difference, YB_difference, Y)
    transformed_image = transformed_flattened.reshape(32, 32, 3)

    # Prepare the output image array
    recovered_image = np.zeros((32, 32, 3), dtype=np.uint8)

    # Extract components
    Illumination = transformed_image[:, :, 0]
    RG_difference = transformed_image[:, :, 1]
    YB_difference = transformed_image[:, :, 2]
    RG_diff = RG_difference * 2 - 1  # between -1 and 1, if negative, Y = R, if positive Y = G
    I = 3 * Illumination  # illumination between 0 and 3
    YB_diff = YB_difference * 2 - 1  # between -1 and 1
    def relu(x):
        return np.maximum(x, 0)

    # 3 * R = (RG_diff >= 0) * (I + 2 * RG_diff + YB_diff) + (RG_diff < 0) * (I + RG_diff + YB_diff)
    # Above eqn is more efficiently implemented as:
    R = (I + RG_diff + YB_diff + relu(RG_diff)) / 3  # Y + relu(RG_diff) = R
    G = - RG_diff + R
    Y = np.minimum(R, G)
    B = - YB_diff + Y
    recovered_image[:, :, 0] = R * 255
    recovered_image[:, :, 1] = G * 255
    recovered_image[:, :, 2] = B * 255

    return recovered_image.astype(np.uint8)


def transform_cifar_representation(cifar_input: np.ndarray) -> np.ndarray:
    """
    Transforms the CIFAR image representation using the specified transformation and returns the flattened version,
    including the extra Y dimension.

    :param cifar_input: A flattened numpy array representing a CIFAR image (originally 32x32x3).
    :return: A transformed and flattened numpy array representation of the CIFAR image, including the Y dimension.
    """
    # Reshape the flattened CIFAR input back into its original 32x32x3 shape
    image = cifar_input.reshape(32, 32, 3)

    # Extract the R, G, B channels
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)

    # Calculate the transformation components
    Illumination = ((R + G + B) / (3.0 * 255.0))

    RG_difference = ((R - G + 255.0) / (2.0 * 255.0))
    Y = np.minimum(R, G).astype(np.float32)
    YB_difference = ((Y - B + 255.0) / (2.0 * 255.0))

    # Combine the components into a transformed image, now including Y
    transformed_image = np.stack((Illumination, RG_difference, YB_difference), axis=-1)

    # Flatten the transformed image into a 1D array
    transformed_flattened = transformed_image.flatten()

    return transformed_flattened


# Load CIFAR-10 data
cifar_raw_data, labels = load_cifar10()  # CIFAR-10
randomized_cifar = randomize_arrays_by_class(cifar_raw_data, labels, 10)
cifar_input = [np.array(sub_array).flatten() for sub_array in randomized_cifar]
# # Transform the CIFAR input
transformed_cifar_input = [transform_cifar_representation(image) for image in cifar_input]

for i in range(len(transformed_cifar_input)):
    # Reverse transform the first image
    test_image = reverse_transform_cifar_representation(transformed_cifar_input[i])
    plot_results(cifar_input[i].reshape(32, 32, 3), test_image.reshape(32, 32, 3))
    diff = cifar_input[i].reshape(32, 32, 3) - test_image.reshape(32, 32, 3)
    print(f"Red: {diff[:, :, 0]}")
    print(f"Green: {diff[:, :, 1]}")
    print(f"Blue: {diff[:, :, 2]}")