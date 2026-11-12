"""A pre-trained sentence model from Hugging Face. train.py uses it by default; --no-mpnet turns it off. """
from __future__ import annotations

from functools import lru_cache

import numpy as np

REPO_ID = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"  # a fixed version, so every download is identical
DIM = 768


class SentenceEncoder:
    def __init__(self, model):
        self._model = model
        self._vectors = {}   # sentence -> its 768 numbers

    def sentence_vectors(self, texts: list[str]) -> np.ndarray:
        """One row of 768 numbers per sentence."""
        self._encode_new(texts)
        rows = []
        for text in texts:
            rows.append(self._vectors[text])
        return np.stack(rows)

    def _encode_new(self, texts: list[str]) -> None:
        """Encode the sentences not seen yet. Tuning asks for the same ones thousands of times."""
        unseen_texts = []
        for text in texts:
            if text not in self._vectors:
                unseen_texts.append(text)
        if not unseen_texts:
            return
        for text, vector in zip(unseen_texts, self._model.encode(unseen_texts, show_progress_bar=False)):
            self._vectors[text] = vector


@lru_cache(maxsize=1)
def english_words() -> frozenset[str]:
    """The whole words in mpnet's word list (about 30,000 English words: 'candle', 'honey', 'sure', ...).

    spelling.py uses it to tell a real word from a typo. Loads only the small tokenizer files, not the model.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    words = set()
    for token in tokenizer.get_vocab():
        if token.isalpha():  # skips word pieces ('##ing'), numbers, punctuation and special tokens ('<mask>')
            words.add(token)
    return frozenset(words)


@lru_cache(maxsize=1)
def load_mpnet() -> SentenceEncoder:
    """Download mpnet (first time only) and open it. Later calls reuse the same copy."""
    import transformers
    from sentence_transformers import SentenceTransformer

    transformers.logging.disable_progress_bar()  # keeps "Loading weights" bars out of the chat
    return SentenceEncoder(SentenceTransformer(REPO_ID, revision=REVISION))
