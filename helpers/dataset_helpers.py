import torch
from typing import List, Tuple

# Transformation functions

#def transform_data(sensory_input: List[torch.Tensor]) ->Tuple[List[torch.Tensor], List[float], List[float]]:
def transform_data(sensory_input: List[torch.Tensor]) -> List[torch.Tensor]:

    normalized_sensory_input = []
    # means = []
    # stds = []
    for tensor in sensory_input:
        mean = torch.mean(tensor).item()
        std = torch.std(tensor).item()
        norm_input = ((tensor - mean) / (std +1e-7))  #epsilon for numerical stability
        # print(f'mean:{mean}, std:{std}, tensor{tensor}, normalized:{norm_input}')
        normalized_sensory_input.append(norm_input)
        # means.append(mean)
        # stds.append(std)
    # return normalized_sensory_input, means, stds
    return normalized_sensory_input


def transform_single_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    mean = torch.mean(tensor).item()
    std = torch.std(tensor).item()
    norm_tensor = (tensor - mean) / (std + 1e-7)
    return norm_tensor, mean, std

def denormalize_data(normalized_data: List[torch.Tensor], means: List[float], stds: List[float]) -> List[torch.Tensor]:
    original_data = []
    for tensor, mean, std in zip(normalized_data, means, stds):
        original_tensor = (tensor * std) + mean  # Reverse the normalization
        original_data.append(original_tensor)
    return original_data

def transform_data_whole_file(sensory_input: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float, float]:
    all_sensory_input = torch.cat([img.clone().flatten() for img in sensory_input])
    mean = torch.mean(all_sensory_input).item()
    std = torch.std(all_sensory_input).item()
    sensory_input = [(img - mean) / std for img in sensory_input]
    return sensory_input, mean, std


#Not needed for 1D
# def get_data(data: str, N: int) -> List[torch.Tensor]:
#     sensory_input = []
#
#     if data == 'MNIST':
#         mnist_raw_data, mnist_labels = load_mnist()  # Implement `load_mnist()`
#         mnist_data = randomize_arrays_by_class(mnist_raw_data, mnist_labels, N)
#         sensory_input = [torch.tensor(img).flatten() for img in mnist_data]
#
#     elif data == 'fashionMNIST':
#         fashion_data, fashion_labels = load_fashion_mnist()  # Implement `load_fashion_mnist()`
#         fashion_mnist_data = randomize_arrays_by_class(fashion_data, fashion_labels, N)
#         sensory_input = [torch.tensor(img).flatten() for img in fashion_mnist_data]
#
#     elif data == 'CIFAR':
#         cifar_raw_data, labels = load_cifar100()  # Implement `load_cifar100()`
#         randomized_cifar = randomize_arrays_by_class(cifar_raw_data, labels, N)
#         sensory_input = [torch.tensor(img).flatten() for img in randomized_cifar]
#
#     return sensory_input

def randomize_arrays_by_class(images: torch.Tensor, labels: torch.Tensor, N: int) -> List[torch.Tensor]:
    num_classes = len(torch.unique(labels))
    class_images = {i: images[labels == i] for i in range(num_classes)}

    for i in class_images:
        perm = torch.randperm(class_images[i].size(0))
        class_images[i] = class_images[i][perm]

    output_images = []
    class_index = 0

    while len(output_images) < N:
        if len(class_images[class_index]) > 0:
            output_images.append(class_images[class_index][0])
            class_images[class_index] = class_images[class_index][1:]
        class_index = (class_index + 1) % num_classes

    return output_images

# CIFAR Representation Transformations

def transform_cifar_representation(cifar_input: torch.Tensor) -> torch.Tensor:
    image = cifar_input.reshape(32, 32, 3).float()

    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    Illumination = (R + G + B) / (3.0 * 255.0)
    RG_difference = (R - G + 255.0) / (2.0 * 255.0)
    Y = torch.minimum(R, G)
    YB_difference = (Y - B + 255.0) / (2.0 * 255.0)

    transformed_image = torch.stack((Illumination, RG_difference, YB_difference), dim=-1)
    return transformed_image.flatten()

def reverse_transform_cifar_representation(transformed_flattened: torch.Tensor) -> torch.Tensor:
    transformed_image = transformed_flattened.reshape(32, 32, 3)

    Illumination = transformed_image[:, :, 0]
    RG_difference = transformed_image[:, :, 1]
    YB_difference = transformed_image[:, :, 2]

    RG_diff = RG_difference * 2 - 1
    I = 3 * Illumination
    YB_diff = YB_difference * 2 - 1

    R = (I + RG_diff + YB_diff + torch.relu(RG_diff)) / 3
    G = -RG_diff + R
    Y = torch.minimum(R, G)
    B = -YB_diff + Y

    recovered_image = torch.stack((R, G, B), dim=-1) * 255
    return recovered_image.byte()

# def load_cifar100() -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Loads CIFAR-10 dataset and normalizes the image arrays.
#
#     :return: numpy arrays of CIFAR-10 normalized image arrays. Size is (10000, 32, 32, 3) of both arrays.
#     """
#     # Load CIFAR-10 dataset
#     (images, labels), (_, _) = cifar100.load_data()
#     images = images / 255.0 # Normalize images
#
#     return images, labels


# def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Load the Fashion MNIST dataset.
#
#     :return: A tuple with Fashion MNIST image vectors (size: (60000, 28, 28)) and their labels (60000, 1).
#     """
#     # Load the Fashion MNIST dataset
#     (mnist_image, labels), _ = fashion_mnist.load_data()
#
#     # Preprocess the data: Normalize the pixel values to be between 0 and 1
#     mnist_dataset = mnist_image.astype('float32') / 255.0
#
#     return mnist_dataset, labels

#
# def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Load and normalize the MNIST dataset.
#
#     :return: A tuple with MNIST image vectors (size: (60000, 28, 28)) and their labels (60000, 1).
#     """
#     # Load the MNIST dataset
#     (mnist_image, labels), _ = mnist.load_data()
#
#     mnist_dataset = mnist_image.astype('float32') / 255.0
#
#     return mnist_dataset, labels
