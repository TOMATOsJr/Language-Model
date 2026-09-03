"""
Datasets and Data Loaders for Nepali Language Modeling.
"""

from pathlib import Path

import numpy as np
# from sympy import re
import torch
from torch.utils.data import Dataset, DataLoader

class TokenStreamDataset(Dataset):
    """
    Dataset over a continuous uint16 token stream.

    Each item contains context_length input tokens and their corresponding
    next-token labels.
    """

    def __init__(
            self,
            token_path: Path,
            context_length: int,
            stride: int | None = None,
    ) -> None:
        self.token_path = token_path
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

        if not self.token_path.is_file():
            raise FileNotFoundError(f"Token file not found: {self.token_path}")

        self.tokens = np.memmap(
            self.token_path,
            dtype=np.uint16,
            mode='r',
        )

        # As we will be taking the context length number of tokens as input
        # its offset by one as its labels.
        self.window_size = context_length + 1

        if len(self.tokens) < self.window_size:
            raise ValueError(
                f"{self.token_path} contains {len(self.tokens)} tokens, "
                f"but at least {self.window_size} are required."
            )

    def __len__(self):
        self.num_examples = (len(self.tokens) - self.window_size) // self.stride + 1
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.__len__():
            raise IndexError(f"Index {index} out of range for dataset of length {self.__len__()}")

        start = index * self.stride
        end = start + self.window_size
        window = self.tokens[start:end]
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        labels = torch.tensor(window[1:], dtype=torch.long)
        return input_ids, labels

def create_dataloader(
    token_path: str | Path,
    context_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    stride: int | None = None,
) -> DataLoader:
    """Create a DataLoader for a uint16 token stream."""
    dataset = TokenStreamDataset(
        token_path=Path(token_path),
        context_length=context_length,
        stride=stride,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )
