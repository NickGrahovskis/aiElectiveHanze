"""Step 1a: turn text into numbers (token IDs) with a vocabulary built from YOUR training sentences."""

from __future__ import annotations

import re

from .data import clean_text

_WORD_PATTERN = re.compile(r"[a-z0-9']+")
PIECE_LENGTHS = (3, 4, 5)  # letters per word piece


def words(text: str) -> list[str]:
    """'I burned my hand!' -> ['i', 'burned', 'my', 'hand']"""
    return _WORD_PATTERN.findall(clean_text(text).lower())


def features(text: str, char_ngrams: bool) -> list[str]:
    """All features of a sentence: its words, its word pairs and (if char_ngrams) its word pieces."""
    tokens = words(text)
    result = _word_features(tokens) + _pair_features(tokens)
    if char_ngrams:
        result += _piece_features(tokens)
    return result


def _word_features(tokens: list[str]) -> list[str]:
    """['my', 'hand'] -> ['w:my', 'w:hand']"""
    result = []
    for word in tokens:
        result.append(f"w:{word}")
    return result


def _pair_features(tokens: list[str]) -> list[str]:
    """['burned', 'my', 'hand'] -> ['p:burned_my', 'p:my_hand']"""
    result = []
    for first, second in zip(tokens, tokens[1:]):
        result.append(f"p:{first}_{second}")
    return result


def _piece_features(tokens: list[str]) -> list[str]:
    result = []
    for word in tokens:
        result += _pieces_of_word(word)
    return result


def _pieces_of_word(word: str) -> list[str]:
    """'hot' -> '<hot>' -> ['c:<ho', 'c:hot', 'c:ot>', 'c:<hot', 'c:hot>', 'c:<hot>']

    < and > mark the start and end of the word, so "<ho" means "a word starting with ho".
    """
    padded = f"<{word}>"
    pieces = []
    for length in PIECE_LENGTHS:
        for start in range(len(padded) - length + 1):
            pieces.append(f"c:{padded[start:start + length]}")
    return pieces


class Vocabulary:
    """Gives every feature seen in the training sentences an ID (its row in the embedding table)."""

    def __init__(self, items: list[str], char_ngrams: bool):
        self.items = items
        self.char_ngrams = char_ngrams
        self._index = {}
        for feature_id, item in enumerate(items):
            self._index[item] = feature_id

    @classmethod
    def build(cls, texts: list[str], char_ngrams: bool) -> "Vocabulary":
        """Collect every feature that appears in the given (training) sentences."""
        seen = set()
        for text in texts:
            seen.update(features(text, char_ngrams))
        return cls(sorted(seen), char_ngrams)

    def __len__(self) -> int:
        return len(self.items)

    def encode(self, text: str) -> list[int]:
        """Sentence -> IDs of its features. Features never seen in training are skipped."""
        feature_ids = []
        for feature in features(text, self.char_ngrams):
            if feature in self._index:
                feature_ids.append(self._index[feature])
        return feature_ids

    def encode_all(self, texts: list[str]) -> list[list[int]]:
        encoded_sentences = []
        for text in texts:
            encoded_sentences.append(self.encode(text))
        return encoded_sentences

    def known_words(self) -> list[str]:
        """Every word that appeared in the training sentences."""
        result = []
        for item in self.items:
            if item.startswith("w:"):
                result.append(item[2:])
        return result

    def known_pairs(self) -> list[tuple[str, str]]:
        """Every two-word pair that appeared in the training sentences: ('my', 'hand'), ..."""
        result = []
        for item in self.items:
            if item.startswith("p:"):
                first, second = item[2:].split("_")
                result.append((first, second))
        return result

    def known_word_fraction(self, text: str) -> float:
        """Share of the message's words that appeared in the training sentences (0.0 - 1.0)."""
        tokens = words(text)
        if not tokens:
            return 0.0
        known = 0
        for word in tokens:
            if f"w:{word}" in self._index:
                known += 1
        return known / len(tokens)
