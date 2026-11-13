from __future__ import annotations

import re
from collections import Counter

from .data import clean_text
from .vocabulary import Vocabulary

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")  # the same words as vocabulary.words(), before lowercasing
MIN_LENGTH = 3     # shorter words are never changed
MAX_DISTANCE = 1   # a typo may be at most this many letters off from the training word


class SpellingFixer:
    def __init__(self, vocabulary: Vocabulary, english_words: frozenset[str]):
        self._english_words = english_words
        self._words = set(vocabulary.known_words())
        self._pairs = set(vocabulary.known_pairs())
        # How common a word is: in how many different training word pairs it appears.
        self._commonness = Counter()
        for first, second in self._pairs:
            self._commonness[first] += 1
            self._commonness[second] += 1

    def fix(self, text: str) -> str:
        """The message with its typos fixed. Returned unchanged if nothing needed fixing."""
        original = text
        text = clean_text(text)
        tokens = list(_TOKEN_PATTERN.finditer(text))
        fixed = []      # (start, end, new word) for every part of the text that changes
        previous_word = None
        position = 0
        while position < len(tokens):
            token = tokens[position]
            word = token.group().lower()
            next_token = None
            if position + 1 < len(tokens):
                next_token = tokens[position + 1]

            joined = self._joined(text, token, next_token)
            if joined is not None:
                fixed.append((token.start(), next_token.end(), _same_case(token.group(), joined)))
                previous_word = joined
                position += 2
                continue

            next_word = None
            if next_token is not None:
                next_word = next_token.group().lower()
            closest = self._closest_word(word, previous_word, next_word)
            if closest != word:
                fixed.append((token.start(), token.end(), _same_case(token.group(), closest)))
            previous_word = closest
            position += 1

        if not fixed:
            return original
        return _replace_parts(text, fixed)

    def _joined(self, text: str, token: re.Match, next_token: re.Match | None) -> str | None:
        """'cho' + 'king' -> 'choking', when the two are separated by one space, at least one of them
        is unknown, and together they form a training word. Otherwise None."""
        if next_token is None or text[token.end():next_token.start()] != " ":
            return None
        first = token.group().lower()
        second = next_token.group().lower()
        if first in self._words and second in self._words:
            return None
        if first + second in self._words:
            return first + second
        return None

    def _closest_word(self, word: str, previous_word: str | None, next_word: str | None) -> str:
        """The training word 1 letter off that fits a neighbour, or the word itself if there is none."""
        if word in self._words or word in self._english_words:
            return word  # a training word, or a real word training never used: not a typo
        if len(word) < MIN_LENGTH or not word.isalpha():
            return word

        best = word
        best_score = None
        for candidate in self._words:
            if abs(len(candidate) - len(word)) > MAX_DISTANCE:
                continue  # too different in length to be close: skip the slow comparison
            if edit_distance(word, candidate) > MAX_DISTANCE:
                continue
            fits_neighbours = ((previous_word, candidate) in self._pairs) + ((candidate, next_word) in self._pairs)
            if fits_neighbours == 0:
                continue  # never seen next to these words in training: too risky to swap in
            # Lower is better: fitting more neighbours first, then most common, then A-Z.
            score = (-fits_neighbours, -self._commonness[candidate], candidate)
            if best_score is None or score < best_score:
                best = candidate
                best_score = score
        return best


def edit_distance(first: str, second: str) -> int:
    """How many letters to add, remove, change or swap to turn one word into the other.

    'birn' -> 'burn' = 1 (change i to u),  'hnad' -> 'hand' = 1 (swap n and a),  'brokem' -> 'broken' = 1
    """
    # row[j] = distance between the first i letters of `first` and the first j letters of `second`
    row_before_previous = None
    previous_row = list(range(len(second) + 1))
    for i in range(1, len(first) + 1):
        row = [i] + [0] * len(second)
        for j in range(1, len(second) + 1):
            change_cost = 0 if first[i - 1] == second[j - 1] else 1
            row[j] = min(previous_row[j] + 1,                  # remove a letter
                         row[j - 1] + 1,                       # add a letter
                         previous_row[j - 1] + change_cost)    # change a letter (or keep it)
            swapped = i > 1 and j > 1 and first[i - 1] == second[j - 2] and first[i - 2] == second[j - 1]
            if swapped:
                row[j] = min(row[j], row_before_previous[j - 2] + 1)  # swap two neighbouring letters
        row_before_previous = previous_row
        previous_row = row
    return previous_row[-1]


def _same_case(original: str, replacement: str) -> str:
    """Keep a capital first letter: ('Hed', 'head') -> 'Head'."""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_parts(text: str, parts: list[tuple[int, int, str]]) -> str:
    """Put each new word in place of text[start:end]; parts are in order and do not overlap."""
    result = []
    position = 0
    for start, end, new_word in parts:
        result.append(text[position:start])
        result.append(new_word)
        position = end
    result.append(text[position:])
    return "".join(result)
