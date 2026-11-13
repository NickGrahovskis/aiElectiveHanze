"""The classifier the chatbot uses: it loads the model that train.py saved and predicts labels with it."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .data import PROJECT_DIR
from .model import load_previous_model
from .mpnet import english_words, load_mpnet
from .spelling import SpellingFixer
from .training import TrainedModel, predict_proba
from .vocabulary import Vocabulary

DEFAULT_CHECKPOINT = PROJECT_DIR / "artifacts" / "model.pt"


@dataclass(frozen=True)
class Prediction:
    ranked: list[tuple[str, float]]     # (label, probability), most likely first
    known_words: float | None = None    # share of the message's words seen during training, if known

    @property
    def label(self) -> str:
        return self.ranked[0][0]

    @property
    def confidence(self) -> float:
        return self.ranked[0][1]


class Classifier(ABC):
    labels: list[str]

    @abstractmethod
    def predict(self, text: str) -> Prediction:
        """Classify a free-text description into one of the Sheet2 labels."""

    def fix_spelling(self, text: str) -> str:
        """The message with its typos fixed. This default changes nothing."""
        return text


class LocalClassifier(Classifier):
    """The network trained from scratch by train.py."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"No trained model at {checkpoint_path}. Run:  python train.py")
        network, checkpoint = load_previous_model(Path(checkpoint_path))
        vocabulary = Vocabulary(checkpoint["vocabulary"], checkpoint["hyperparams"]["char_ngrams"])
        mpnet = None
        if checkpoint["hyperparams"].get("mpnet", False):  # the network was trained on mpnet vectors
            mpnet = load_mpnet()
        self._model = TrainedModel(network, vocabulary, mpnet)
        self.labels = checkpoint["labels"]
        self._spelling = None  # made on first use, so --no-fix-spelling never loads mpnet's word list

    def predict(self, text: str) -> Prediction:
        probabilities = predict_proba(self._model, [text])[0]
        ranked = _rank_labels(self.labels, probabilities)
        return Prediction(ranked, self._model.vocabulary.known_word_fraction(text))

    def fix_spelling(self, text: str) -> str:
        """Typos replaced by words from the training sentences (see spelling.py)."""
        if self._spelling is None:
            self._spelling = SpellingFixer(self._model.vocabulary, english_words())
        return self._spelling.fix(text)


def _rank_labels(labels: list[str], probabilities) -> list[tuple[str, float]]:
    """Pair every label with its probability, most likely first."""
    most_likely_first = probabilities.argsort()[::-1]
    ranked = []
    for label_number in most_likely_first:
        ranked.append((labels[label_number], float(probabilities[label_number])))
    return ranked
