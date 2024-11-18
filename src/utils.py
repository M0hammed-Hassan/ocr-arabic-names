import cv2
import json
import torch
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence


def calculate_average_resolution(data_path: Path) -> Tuple[float, float]:
    """
    Calculate the average resolution of all PNG images in a given directory.

    Parameters:
        data_path (Path): The path to the directory containing the PNG images.

    Returns:
        tuple: A tuple (height, width) representing the average resolution of the images.
        If no PNG images are found, returns (0, 0).
    """
    resolution = np.empty((0, 2), int)
    for file in data_path.glob("*.png"):
        image = cv2.imread(str(file))
        h, w, _ = image.shape
        resolution = np.vstack((resolution, np.array([h, w])))

    if resolution.size == 0:
        return (0.0, 0.0)

    return resolution.mean(axis=0)


def preprocess_image(image_path: str, target_size: Tuple[int, int]) -> np.array:
    """
    Preprocess an image by reading it from a file, converting it to grayscale, resizing it, and normalizing pixel values.

    Parameters:
        image_path (str): The str to the image file.
        target_size (tuple): The desired size (width, height) for the resized image.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array with pixel values normalized to the range [0, 1].
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image


def load_dataset(
    dataset_dir: Path, target_size: Tuple[int, int]
) -> Tuple[List[np.array], List[str], List[str]]:
    """
    Load a dataset of images and their corresponding labels from a directory.

    Parameters:
        dataset_dir (Path): The path to the directory containing the images and labels.
        target_size (tuple): The desired size (width, height) for the resized images.

    Returns:
        tuple: A tuple (images, labels, charset) containing the preprocessed images, their corresponding labels, and the charset used for label encoding.
    """
    images = []
    labels = []
    charset = set()

    for file in Path(dataset_dir).glob("*.png"):
        image_path = file
        label_path = dataset_dir / f"{file.stem}.gt.txt"

        with open(str(label_path), "r") as label_file:
            label = label_file.read().strip()

        charset.update(label)
        images.append(preprocess_image(str(image_path), target_size))
        labels.append(label)

    charset = sorted(charset)
    return (np.array(images), labels, charset)


def setup_logs() -> None:
    """
    Set up logging for the application.

    This function creates a new log file in the 'logs' directory with a unique name based on the current date and time.
    The log file is rotated every 10 MB.

    Parameters:
        None

    Returns:
        None
    """
    run_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.add(logs_dir / f"{run_date}.log", rotation="10 MB")
    logger.info("Logging Started")


def create_char_to_idx(charset: dict) -> dict:
    """
    Create a dictionary mapping characters to indices.

    Parameters:
        charset (List[str]): The character set.

    Returns:
        dict: A dictionary mapping each character to a unique index.
    """
    return {char: idx for idx, char in enumerate(charset)}


def save_idx_to_char(char_to_idx: dict) -> dict:
    """
    Create a dictionary mapping indices to characters.

    Parameters:
        char_to_idx (dict): A dictionary mapping each character to a unique index.

    Returns:
        dict: A dictionary mapping each index to its corresponding character.
    """
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    with open("idx_to_char.json", "w") as json_file:
        json.dump(idx_to_char, json_file)
    logger.info("Json file for indices to char saved successfully")


def collate_fn(batch):
    """
    Custom collate function to handle variable-length labels.
    Pads labels to the same length.
    """
    images, labels = zip(*batch)
    images = torch.stack(images)

    labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return images, labels


def train_step(
    train_data_loader: torch.tensor,
    model: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn,
    device: torch.device,
) -> float:
    """
    Perform a single training step for a sequence recognition model.

    This function iterates over a training data loader, moves the data to the specified device,
    makes predictions using the model, calculates the loss, updates the model parameters,
    and returns the average loss for the entire training step.

    Parameters:
        train_data_loader (torch.tensor): A data loader containing batches of training images and labels.
        model (torch.nn.Module): The sequence recognition model to be trained.
        optimizer (torch.optim): The optimizer used to update the model parameters.
        criterion (torch.nn): The loss function to be used for training.
        device (torch.device): The device (CPU or GPU) where the computations will be performed.

    Returns:
        float: The average loss for the entire training step.
    """
    model.train()
    running_loss = 0.0
    loop = tqdm(train_data_loader)
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        log_probs = torch.nn.functional.log_softmax(predictions, dim=2)
        input_length = torch.full(
            (images.size(0),),
            log_probs.size(1),
            dtype=torch.long,
            device=device,
        )
        target_length = torch.sum(labels != -1, dim=1)
        flattened_labels = labels[labels != -1]
        log_probs = log_probs.permute(1, 0, 2)

        loss = criterion(log_probs, flattened_labels, input_length, target_length)
        loop.set_postfix(train_loss=loss.item())
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss / len(train_data_loader)


def save_checkpt(checkpt: torch.nn.Module, checkpt_name: str) -> None:
    """
    Save a checkpoint of a trained model to the 'checkpoints' directory.

    This function creates a 'checkpoints' directory if it doesn't exist, and then saves the provided checkpoint
    under the specified name in this directory.

    Parameters:
        checkpt (torch.nn.Module): The trained model checkpoint to be saved.
        checkpt_name (str): The name of the checkpoint file.

    Returns:
        None
    """
    checkpt_dir = Path("checkpoints")
    checkpt_dir.mkdir(exist_ok=True)
    torch.save(checkpt, checkpt_dir / checkpt_name)
