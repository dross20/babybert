from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


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


class WordPieceTokenizer:
    """Basic implementation of WordPiece tokenizer."""

    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab = []

    @staticmethod
    def _pretokenize(text: str) -> List[str]:
        """
        Perform pretokenization on a text. Carries out the following steps:

        - Converts the text to lowercase.
        - Splits on and removes whitespaces.
        - Splits on and preserves punctuation.

        Args:
            text: The text to pretokenize.
        Returns:
            A list containing each part of the pretokenized text.

        Example:
            >>> WordPieceTokenizer._pretokenize("Hello, world!")
            ["hello", ",", "world", "!"]
        """
        text_lower = text.lower()
        parts = re.split(r"\s|([.!?,;])", text_lower)
        return [part for part in parts if part]

    @staticmethod
    def _score_pair(
        first_token: str, second_token: str, pair_freq: int, token_freqs: Dict[str, int]
    ) -> float:
        """
        Computes the WordPiece score of a pair of tokens using the following
        formula:

        score = frequency(pair) / (frequency(first) * frequency(second))

        Args:
            first_token: The first token in the pair.
            second_token: The second token in the pair.
            pair_freq: The frequency at which the pair occurs in the corpus.
            token_freqs: The dictionary containing token to frequency mappings.
        Returns:
            The WordPiece pair score, which is the pair's frequency normalized by the
            product of the individual token frequencies.
        """
        return pair_freq / (token_freqs[first_token] * token_freqs[second_token])

    @staticmethod
    def _compute_pair_scores(tokenized: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Compute the WordPiece score for each consecutive pair of tokens in the corpus.

        Args:
            tokenized: The list of tokens.
        Returns:
            A dictionary, containing a mapping between consecutive token pairs and
            their scores.
        """
        token_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)

        for tokens in tokenized:
            for token in tokens:
                token_freqs[token] += 1
            for pair in zip(tokens, tokens[1:]):
                pair_freqs[pair] += 1

        scores = {
            pair: WordPieceTokenizer._score_pair(*pair, freq, token_freqs)
            for pair, freq in pair_freqs.items()
        }
        return scores

    @staticmethod
    def _merge_tokens(
        pair: Tuple[str, str], replacement: str, words: List[List[str]]
    ) -> Dict[str, List[str]]:
        """
        Finds instances of the specified pair of consecutive tokens and replaces them
        with a new token.

        Args:
            pair: The pair of consecutive tokens to find and replace.
            replacement: The token with which to replace the pair of tokens.
            words: The words in which to replace the tokens.
        Returns:
            The list of words, with all instances of the target pair merged.

        Example:
            >>> WordPieceTokenizer._merge_tokens(
                ("a", "##b"),
                "ab",
                [["a", "##b", "##s"], ["a", "##r", "##c"], ["a", "##b", "##late"]]
            )
            [["ab", "##s"], ["a", "##r", "##c"], ["ab", "##late"]]
        """
        for tokens in words:
            i = 0
            while i < len(tokens) - 1:
                current = (tokens[i], tokens[i + 1])
                if current == pair:
                    tokens.pop(i + 1)
                    tokens[i] = replacement
                i += 1
        return words

    @staticmethod
    def _make_replacement(a: str, b: str) -> str:
        """
        Make a merged token from a pair of tokens.

        Args:
            a: The first token to merge.
            b: The second token to merge.
        Returns:
            The token created by merging.

        Example:
            >>> WordPieceTokenizer._make_replacement("h", "##i")
            "hi"
        """
        return a + (b[2:] if b.startswith("##") else b)

    @staticmethod
    def _get_character_tokens(word: str):
        """
        Split a word into its component character tokens, using the WordPiece style.
        This includes a "##" prefix before each non-leading character.

        Args:
            word: The word to split.
        Returns:
            A list of character tokens made from splitting the input.

        Examples:
            >>> WordTokenizer._get_character_tokens("banana")
            ["b", "##a", "##n", "##a", "##n", "##a"]
        """
        return [
            character if i == 0 else f"##{character}"
            for i, character in enumerate(word)
        ]

    def train(self, corpus: List[str]) -> None:
        """
        Trains a tokenizer using the WordPiece protocol. The training comprises the
        following steps:

        1. Pretokenizes the corpus, performing basic operations such as splitting on
        whitespace and punctuation.
        2. Converts each word in the corpus to character tokens. These character tokens
        make up the initial state of our vocabulary.
        3. Computes a score for each pair of consecutive tokens in the corpus, denoting
        how likely they are to occur next to each other.
        4. Selects the pair with the highest score to merge, replacing instances of the
        component tokens with the merged token and adding the merged token to the
        vocabulary.
        5. Repeats steps 3 and 4 until the desired vocabulary size is reached, or no
        more merges can be made (meaning each entire word in the corpus is in the
        vocabulary.)

        Args:
            corpus: The list of texts to use for training. Should contain a
            diverse set of words and punctuation marks.
        """
        pretokenized_words = [
            word for text in corpus for word in WordPieceTokenizer._pretokenize(text)
        ]

        word_tokens = [
            WordPieceTokenizer._get_character_tokens(word)
            for word in pretokenized_words
        ]

        self.vocab = list(set(token for word in word_tokens for token in word))
        self.vocab.extend(self.config.special_tokens)

        while len(self.vocab) < self.config.target_vocab_size:
            scores = self._compute_pair_scores(word_tokens)

            if not scores:
                warnings.warn(
                    "No more pairs left to merge, "
                    f"stopping training with {len(self.vocab)} tokens in vocabulary."
                )
                break

            merge_pair = max(scores, key=scores.get)
            replacement = self._make_replacement(*merge_pair)
            self.vocab.append(replacement)
            word_tokens = self._merge_tokens(merge_pair, replacement, word_tokens)

    @staticmethod
    def _make_all_prefix_substrings(string: str) -> List[str]:
        """
        Generates all prefix substrings from a given string.
        Utility function for `_encode_word`.

        Args:
            string: The string for which to generate the substrings.
        Returns:
            The list of all prefix substrings possible.

        Examples:
            >>> WordPieceTokenizer._make_all_prefix_substrings("garage")
            ["garage", "garag", "gara", "gar", "ga", "g"]
        """
        return [string[:-i] if i != 0 else string for i in range(len(string) + 1)]

    def _encode_word(self, word: str) -> List[str]:
        """
        Encodes a word using a longest-first matching strategy. Uses tokens from the
        vocabulary to encode the word.

        Args:
            word: The word to encode.
        Returns:
            The word encoded as tokens from the vocabulary.
        """
        vocab = self.vocab
        tokens = []

        while word:
            for substring in WordPieceTokenizer._make_all_prefix_substrings(word):
                if substring in vocab:
                    tokens.append(substring)
                    if len(substring) == len(word):
                        return tokens
                    else:
                        word = f"##{word[len(substring) :]}"
                        break
                elif not substring:
                    return [self.config.unknown_token]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a text, converting it into a set of tokens contained in the
        tokenizer's vocabulary.

        Args:
            text: The text to tokenize.
        Returns:
            A list of tokens generated from encoding the input text.
        """
        pretokenized_words = WordPieceTokenizer._pretokenize(text)
        encoded = [
            token for word in pretokenized_words for token in self._encode_word(word)
        ]
        return encoded

    def encode(self, text: str) -> List[int]:
        """
        Encodes an text as a list of token IDs.

        Args:
            text: The text to encode.
        Returns:
            The token IDs of the tokenized text.
        """
        tokens = self.tokenize(text)
        return [self.vocab.index(token) for token in tokens]

    def decode(self, token_ids: str) -> List[str]:
        """
        Decodes a text from a list of token IDs.

        Args:
            token_ids: The token IDs to decode.
        Returns:
            The text decoded from the token IDs.
        """
        return [self.vocab[index] for index in token_ids]

    @property
    def vocab_size(self):
        """The number of tokens in the vocabulary."""
        return len(self.vocab)

    @classmethod
    def from_pretrained(cls, name: str) -> WordPieceTokenizer:
        """
        Load a pretrained tokenizer from a checkpoint.

        Args:
            name: The name of the checkpoint from which to save the model
            (small, base, large).
        Returns:
            A `WordPieceTokenizer` with a pretrained vocabulary and preset
            configuration.
        """
        raise NotImplementedError

    def save_pretrained(self, directory: Path) -> None:
        raise NotImplementedError
