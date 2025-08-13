from collections import defaultdict
from typing import Dict, List, Tuple


class WordPieceTokenizer:
    def _pretokenize(self, word: str) -> List[str]:
        punctuation = ".!?,;"
        output = []
        current = ""
        for character in word:
            if character in punctuation:
                if current:
                    output.append(current)
                    current = ""
                output.append(character)
            else:
                current += character
        if current:
            output.append(current)

        return output

    def _compute_pair_scores(
        self, tokenized: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str], float]:
        token_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)

        for tokens in tokenized.values():
            for token in tokens:
                token_freqs[token] += 1
            if len(tokens) > 1:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freqs[pair] += 1

        scores = {
            pair: pair_freqs[pair] / (token_freqs[pair[0]] * token_freqs[pair[1]])
            for pair in pair_freqs.keys()
        }
        return scores

    def _merge_tokens(
        self, a: str, b: str, replacement: str, words: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        for word, tokens in words.items():
            if len(tokens) >= 1:
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] == a and tokens[i + 1] == b:
                        words[word].pop(i + 1)
                        words[word][i] = replacement
                    i += 1
        return words

    def _make_replacement(self, a, b):
        return a + b[2:] if b.startswith("##") else a + b

    def train(self, corpus: List[str], target_vocab_size: int = 1000) -> None:
        vocab = []

        words = [word for sentence in corpus for word in sentence.split()]
        pretokenized_words = [
            pretokenized for word in words for pretokenized in self._pretokenize(word)
        ]

        words_tokenized = {
            word: [
                character if i == 0 else f"##{character}"
                for i, character in enumerate(word)
            ]
            for word in pretokenized_words
        }

        vocab = set(token for tokens in words_tokenized.values() for token in tokens)

        while len(vocab) < target_vocab_size:
            scores = self._compute_pair_scores(words_tokenized)
            merge_pair = max(scores, key=scores.get)
            replacement = self._make_replacement(merge_pair)
            vocab.add(replacement)
            words_tokenized = self._merge_tokens(
                *merge_pair, replacement, words_tokenized
            )

        self.vocab = vocab

    def _encode_word(self, word: str) -> List[str]:
        vocab = list(self.vocab)
        tokens = []
        while word:
            i = len(word)
            while i > 0 and word[:i] not in vocab:
                print(word[:i])
                i = i - 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            print(tokens)
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(self, text: str) -> Tuple:
        words = [word for word in text.split()]
        pretokenized_words = [
            pretokenized for word in words for pretokenized in self._pretokenize(word)
        ]
        encoded = [self._encode_word(word) for word in pretokenized_words]
        return encoded

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def from_pretrained(cls, name):
        return cls()
