import torch
from model import BabyBERT, BabyBERTForMLM
from torch.utils.data import DataLoader, Dataset


class TrainerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Trainer:
    """Simple trainer class for pretraining BabyBERT using masked language modeling."""

    def __init__(self, model: BabyBERT, config: TrainerConfig):
        self.model = model
        self.config = config
        self.device = config.device

    def run(self, data: Dataset) -> BabyBERT:
        """
        Pretrain the BabyBERT model on the provided dataset using MLM.

        Args:
            data: The `Dataset` to use for training.
        Returns:
            out: The BabyBERT model with updated parameters after training.
        """
        mlm_model = BabyBERTForMLM(self.model).to(self.device)
        config = self.config

        optimizer = torch.optim.Adam(mlm_model.parameters(), self.config.learning_rate)

        loader = DataLoader(
            data,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
        )

        mlm_model.train()

        for batch in loader:
            batch = [sample.to(self.device) for sample in batch]
            x, masks, y = batch
            _, loss = mlm_model(x, mask=masks, labels=y)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), config.max_grad_norm)
            optimizer.step()

        return self.model
