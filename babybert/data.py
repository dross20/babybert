from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from tokenizer import WordPieceTokenizer


class LanguageModelingDataset(Dataset):
    """Class for storing an LM dataset."""

    def __init__(
        self, token_ids: list[list[int]], attention_mask: list[list[int]], labels=None
    ):
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[list[int], list[int], list[int]] | tuple[list[int], list[int]]:
        if self.labels:
            return self.token_ids[index], self.attention_mask[index], self.labels[index]
        else:
            return self.token_ids[index], self.attention_mask[index]

    @classmethod
    def from_dict(cls, data: dict[str, list[int]]) -> LanguageModelingDataset:
        """
        Create a new dataset instance from a dictionary.

        Args:
            data: A dictionary containing entries `token_ids`, `attention_mask`, and
                  optionally `labels`.
        Returns:
            A new dataset object with values populated from the dictionary.
        """
        return cls(
            data.get("token_ids"), data.get("attention_mask"), data.get("labels")
        )


class CollatorForMLM:
    """Data collator for applying masks to batches of tokens."""

    def __init__(self, tokenizer: WordPieceTokenizer, mask_prob: float = 0.1):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __call__(self, batch):
        token_ids, attention_mask = zip(*batch)
        ...
