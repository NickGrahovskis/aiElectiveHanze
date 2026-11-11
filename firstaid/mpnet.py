"""A pre-trained sentence model from Hugging Face. train.py uses it by default; --no-mpnet turns it off.

all-mpnet-base-v2 (Sentence Transformers) is a transformer: a large neural network that learned
English from over a billion sentence pairs. It reads a whole sentence, word order included, and
turns it into 768 numbers that stand for its meaning. Sentences that mean the same thing get similar
numbers even when they share few words: "I burned my hand" and "my hand has a burn". With mpnet on,
those numbers are the sentence embedding, instead of the embeddings learned from your sentences.

These numbers are NOT learned from your sentences: they come from a downloaded model that stays
fixed. The hidden layers on top of them are still trained from scratch.

Source: https://huggingface.co/sentence-transformers/all-mpnet-base-v2 (about 420 MB, downloaded
once and cached by huggingface_hub).
"""
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
def load_mpnet() -> SentenceEncoder:
    """Download mpnet (first time only) and open it. Later calls reuse the same copy."""
    import transformers
    from sentence_transformers import SentenceTransformer

    transformers.logging.disable_progress_bar()  # keeps "Loading weights" bars out of the chat
    return SentenceEncoder(SentenceTransformer(REPO_ID, revision=REVISION))
