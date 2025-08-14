from dataclasses import dataclass, field
from typing import List


class BabyBERTConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TrainerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass
class TokenizerConfig:
    _special_tokens: List[str] = field(
        default_factory=lambda: ["[CLS]", "[PAD]", "[SEP]", "[MASK]"]
    )
    unknown_token: str = "[UNK]"
    target_vocab_size: int = 1000

    @property
    def special_tokens(self):
        return [*self._special_tokens, self.unknown_token]
