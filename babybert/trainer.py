from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    import torch.nn as nn


class TrainerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Trainer:
    """Simple model-agnostic PyTorch trainer class."""

    def __init__(self, model: nn.Module, config: TrainerConfig):
        self.model = model
        self.config = config
        self.device = config.device

    def run(self, data: Dataset):
        """
        Train the BabyBERT model on the provided dataset.

        Args:
            data: The `Dataset` to use for training.
        """
        model = self.model
        config = self.config

        optimizer = torch.optim.Adam(model.parameters(), self.config.learning_rate)

        loader = DataLoader(
            data,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            collate_fn=config.collator,
        )

        model.train()

        for batch in loader:
            batch = [sample.to(self.device) for sample in batch]
            x, masks, y = batch
            _, loss = model(x, mask=masks, labels=y)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
