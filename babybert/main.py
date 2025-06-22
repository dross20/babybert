from config import BabyBERTConfig
from model import BabyBERT
import torch

config = BabyBERTConfig(
    vocab_size=5,
    hidden_size=10,
    block_size=5,
    n_blocks=1,
    n_heads=10,
    sep_token_id=0
)

input = torch.asarray([[1, 2, 0, 4, 3], [1, 2, 0, 3, 4]])
model = BabyBERT(config)
output = model(input)
print(output.shape)