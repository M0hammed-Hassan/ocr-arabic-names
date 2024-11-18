import torch
import multiprocessing
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataDirectories:
    train_dir: Path


@dataclass
class HyperParams:
    epochs: int
    device: str
    batch_size: int
    num_workers: int
    learning_rate: float


data_dirs = DataDirectories(train_dir=Path("data"))
hyperparams = HyperParams(
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    device=torch.device("cuda:0"),
    num_workers=multiprocessing.cpu_count(),
)
