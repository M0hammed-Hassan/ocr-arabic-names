import torch.nn as nn
from typing import Tuple
from loguru import logger
from model import CRNNModel
import torch.optim as optim
from seeding import seed_everything
from dataset import ArabicNamesDataset
from torch.utils.data import DataLoader
from configs import data_dirs, hyperparams
from utils import (
    setup_logs,
    calculate_average_resolution,
    load_dataset,
    create_char_to_idx,
    train_step,
    save_checkpt,
    collate_fn,
    save_idx_to_char,
)


def prepare_data() -> Tuple[DataLoader, int, list]:
    """
    Prepare the training dataset and data loader.

    Returns:
        DataLoader: The data loader for the training dataset.
        int: The number of unique characters in the dataset plus one (for the blank label).
        list: The character set used in the dataset.
    """

    mean_height, mean_width = calculate_average_resolution(data_dirs.train_dir)
    images, labels, charset = load_dataset(
        data_dirs.train_dir, target_size=(int(mean_width), int(mean_height))
    )
    char_to_idx = create_char_to_idx(charset)
    save_idx_to_char(char_to_idx)

    train_dataset = ArabicNamesDataset(images, labels, char_to_idx)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=hyperparams.num_workers,
    )
    logger.info("Data prepared successfully")

    return train_data_loader, len(charset) + 1


def initialize_model(num_classes: int) -> Tuple[nn.Module, nn.CTCLoss, optim.Optimizer]:
    """
    Initializes a CRNN model, loss function, and optimizer for training.

    The function creates an instance of the CRNNModel with the given number of classes,
    moves it to the specified device, initializes a CTC loss function with the blank label,
    and creates an AdamW optimizer with the specified learning rate.

    Parameters:
        num_classes (int): The number of unique characters in the dataset plus one (for the blank label).

    Returns:
        nn.Module: The initialized CRNN model.
        nn.CTCLoss: The initialized CTC loss function.
        torch.optim.Optimizer: The initialized AdamW optimizer.
    """
    model = CRNNModel(num_classes).to(hyperparams.device)
    criterion = nn.CTCLoss(blank=num_classes - 1, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)
    return model, criterion, optimizer


def train_model(
    train_data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> None:
    """
    Trains the CRNN model using the provided training dataset, model, criterion, and optimizer.

    The function iterates over the specified number of epochs, performs a training step for each epoch,
    and saves the best and last checkpoints based on the training loss.

    Parameters:
        - train_data_loader (DataLoader): The data loader for the training dataset.
        - model (nn.Module): The initialized CRNN model.
        - criterion (nn.Module): The initialized CTC loss function.
        - optimizer (optim.Optimizer): The initialized AdamW optimizer.

    Returns:
        - Nones
    """
    min_loss = float("inf")
    for epoch in range(1, hyperparams.epochs + 1):
        train_loss = train_step(
            train_data_loader, model, optimizer, criterion, device=hyperparams.device
        )
        checkpt = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if train_loss < min_loss:
            min_loss = train_loss
            save_checkpt(checkpt, "best.pth.tar")
            logger.info(f"Epoch: {epoch} ==> best checkpt saved")
        logger.info(
            f"Epoch: {epoch} / {hyperparams.epochs} | training loss = {train_loss:.4f}"
        )
    save_checkpt(checkpt, "last.pth.tar")
    logger.info("Last checkot saved.")


def main() -> None:
    """
    The main function orchestrates the training process.

    Parameters:
        None

    Returns:
        None
    """
    train_data_loader, num_classes = prepare_data()
    model, criterion, optimizer = initialize_model(num_classes)
    train_model(train_data_loader, model, criterion, optimizer)


if __name__ == "__main__":
    setup_logs()
    seed_everything()
    main()
