"""Step 1a: turn text into numbers (token IDs) with a vocabulary built from YOUR training sentences.

No pretrained model is used. Each sentence is split into features:
    words            "burned", "my", "hand"
    word pairs       "burned_my", "my_hand"            (keeps a little word order: "not_broken")
    word pieces      "<bu", "bur", "urn", "rne", ...    (3-5 letters, optional)

Word pieces let the model connect words it has never seen in full: "burned" and "burn" share
"<bu", "bur", "urn". Every feature seen in the training sentences gets an ID; unseen features are
ignored. The IDs then go into the network's embedding layer (model.py), which learns a vector of
numbers for each one: those learned vectors are the embeddings.
"""
from __future__ import annotations

import re

from .data import clean_text

_TOKEN = re.compile(r"[a-z0-9']+")
PIECE_LENGTHS = (3, 4, 5)  # letters per word piece


def words(text: str) -> list[str]:
    """'I burned my hand!' -> ['i', 'burned', 'my', 'hand']"""
    return _TOKEN.findall(clean_text(text).lower())


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
        ids = []
        for feature in features(text, self.char_ngrams):
            if feature in self._index:
                ids.append(self._index[feature])
        return ids

    def encode_all(self, texts: list[str]) -> list[list[int]]:
        encoded = []
        for text in texts:
            encoded.append(self.encode(text))
        return encoded

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
