import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.query_embedding = nn.Linear(config.hidden_size, config.hidden_size // config.n_heads)
        self.key_embedding = nn.Linear(config.hidden_size, config.hidden_size // config.n_heads)
        self.value_embedding = nn.Linear(config.hidden_size, config.hidden_size // config.n_heads)

    def forward(self, x, mask=None):
        q = self.query_embedding(x)
        k = self.key_embedding(x)
        v = self.value_embedding(x)

        raw_attention_weights = (q @ torch.transpose(k, -2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1)
            raw_attention_weights = raw_attention_weights.masked_fill(mask == 0, float("-inf"))

        normalized_attention_weights = F.softmax(raw_attention_weights, -1)
        context_vector = normalized_attention_weights @ v

        return context_vector

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention_heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_heads)])
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, mask=None):
        x = torch.cat([attention_head(x, mask) for attention_head in self.attention_heads], dim=-1)
        x = self.output_projection(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mhsa = MultiHeadSelfAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, mask=None):
        x = x + self.mhsa(x, mask)
        x = self.layer_norm_1(x)
        x = x + self.mlp(x)
        x = self.layer_norm_2(x)
        return x


class BabyBERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(2, config.hidden_size)
        self.positional_embeddings = nn.Embedding(config.block_size, config.hidden_size)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])

    def forward(self, x, mask=None):
        position_ids = torch.arange(0, self.config.block_size, device=x.device).unsqueeze(0)

        sep_token_mask = x == self.config.sep_token_id
        sep_token_position = sep_token_mask.int().argmax(dim=1).unsqueeze(1)

        segment_encodings = (position_ids > sep_token_position).long()

        x = self.token_embeddings(x)
        x = x + self.segment_embeddings(segment_encodings)
        x = x + self.positional_embeddings(position_ids)

        for block in self.transformer_blocks:
            x = block(x, mask)

        return x