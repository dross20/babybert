import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """Implementation of a single self-attention head."""

    def __init__(self, config):
        super().__init__()

        # The size of each attention head's latent space is equal to the embedding
        # space's size divided by the number of heads. That way, after we compute the
        # context vector for each attention head, we can concatenate them to produce a
        # vector that's the same size as the original embedding space.
        head_size = config.hidden_size // config.n_heads

        # Layers for projecting the input embeddings into three spaces:
        # the query (Q) space, the key (K) space, and the value (V) space.
        self.query_embedding = nn.Linear(config.hidden_size, head_size)
        self.key_embedding = nn.Linear(config.hidden_size, head_size)
        self.value_embedding = nn.Linear(config.hidden_size, head_size)

        self.dropout = nn.Dropout(config.attention_dropout_probability)

    def forward(self, x, mask=None):
        # Project the inputs into each space.
        q = self.query_embedding(x)
        k = self.key_embedding(x)
        v = self.value_embedding(x)

        # Perform scaled dot product attention (SDPA) using the query and key matrices
        # - this yields the similarity between each embedding in our input sequences.
        raw_attention_weights = (q @ torch.transpose(k, -2, -1)) / math.sqrt(q.size(-1))

        # If we have an attention mask, we replace each of the masked tokens with a
        # negative infinity in our attention weight matrix. This makes sure that the
        # attention score will be 0 when we compute the softmax later.
        if mask is not None:
            mask = mask.unsqueeze(1)
            raw_attention_weights = raw_attention_weights.masked_fill(
                mask == 0, float("-inf")
            )

        # Compute the softmax over each input sequence, normalizing the scores.
        normalized_attention_weights = F.softmax(raw_attention_weights, -1)
        normalized_attention_weights = self.dropout(normalized_attention_weights)

        # Taking the dot product of the weight matrix and the value matrix gives us our
        # final context vector.
        context_vector = normalized_attention_weights @ v

        return context_vector


class MultiHeadSelfAttention(nn.Module):
    """Implementation of a multi-head self-attention layer."""

    def __init__(self, config):
        super().__init__()

        # Initialize n attention heads - this allows us to capture n sets of
        # relationships between token embeddings.
        self.attention_heads = nn.ModuleList(
            [AttentionHead(config) for _ in range(config.n_heads)]
        )
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_projection_dropout_probability)

    def forward(self, x, mask=None):
        # Concatenate the output of each of the n attention heads.
        x = torch.cat(
            [attention_head(x, mask) for attention_head in self.attention_heads], dim=-1
        )

        # Project the concatenated vectors to the output space.
        x = self.output_projection(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Implementation of a transformer block."""

    def __init__(self, config):
        super().__init__()

        self.mhsa = MultiHeadSelfAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)

        # Simple multilayer perceptron (MLP) to handle tokenwise learning.
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.mlp_dropout_probability),
        )
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, mask=None):
        # Pass the input through our multi-head self attention layer.
        x = x + self.mhsa(x, mask)
        x = self.layer_norm_1(x)

        # Pass the input through the MLP.
        x = x + self.mlp(x)
        x = self.layer_norm_2(x)
        return x


class BabyBERT(nn.Module):
    """Minimal implementation of [BERT](https://arxiv.org/pdf/1810.04805)."""

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Learned embeddings for each token ID in the vocabulary.
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Learned embeddings for each segment; we only have segments 1 and 2 in BERT.
        self.segment_embeddings = nn.Embedding(2, config.hidden_size)

        # Learned embeddings for each token position.
        # Having positional embeddings is necessary for the model to understand the
        # order of the input tokens.
        self.positional_embeddings = nn.Embedding(config.block_size, config.hidden_size)

        self.dropout = nn.Dropout(config.embedding_dropout_probability)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_blocks)]
        )

    def forward(self, x, mask=None):
        # Get the position IDs - these range from 0 to the max length of an input
        # sequence (stored in the block_size variable).
        position_ids = torch.arange(
            0, self.config.block_size, device=x.device
        ).unsqueeze(0)

        # Find the location of the [SEP] token. If there's just one segment in the
        # input, [SEP] will occur at the end of the input.
        sep_token_mask = x == self.config.sep_token_id
        sep_token_position = sep_token_mask.int().argmax(dim=1).unsqueeze(1)

        # All positions leading up to and including the [SEP] token belong to segment 0;
        # the rest belong to segment 1.
        segment_encodings = (position_ids > sep_token_position).long()

        # Sum the token, segment, and positional embeddings to obtain the complete
        # embedding tensor.
        x = self.token_embeddings(x)
        x = x + self.segment_embeddings(segment_encodings)
        x = x + self.positional_embeddings(position_ids)

        x = self.dropout(x)

        # Pass the embeddings through each transformer block in our model.
        for block in self.transformer_blocks:
            x = block(x, mask)

        return x
