import torch
from typing import Tuple
from torch.utils.data import Dataset


class ArabicNamesDataset(Dataset):
    def __init__(self, images: list, labels: list, char_to_idx: dict) -> None:
        """
        Initialize an instance of ArabicNamesDataset.

        Parameters:
            images (list): A list of input images. Each image is represented as a 2D array.
            labels (list): A list of corresponding labels for the images. Each label is a string.
            char_to_idx (dict): A dict that maps each character to its corresponding index.

        Returns:
            None
        """
        super(ArabicNamesDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.char_to_idx = char_to_idx

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Parameters:
            None

        Returns:
            int: The total number of images in the dataset.
        """

        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Retrieves a specific image and its corresponding label from the dataset.

        Parameters:
            index (int): The index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the image and label as PyTorch tensors.
            - image (torch.Tensor): The image as a 2D tensor with a single channel, unsqueezed to have a shape of (1, height, width).
            - label (torch.Tensor): The label as a 1D tensor of character indices.
        """
        image = self.images[index]
        label = self.labels[index]
        label_indices = [self.char_to_idx[char] for char in label]
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(
            label_indices, dtype=torch.long
        )
