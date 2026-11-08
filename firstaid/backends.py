"""Classifier backends. The chatbot only talks to the `Classifier` interface, so the local
PyTorch model can later be swapped for (or backed up by) Gemini without touching chat.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .data import PROJECT_DIR

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


class LocalClassifier(Classifier):
    """The network trained from scratch by train.py."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"No trained model at {checkpoint_path}. Run:  python train.py")
        # Imported here so the Gemini backend does not need torch.
        from .model import load_checkpoint
        from .training import TrainedModel, predict_proba
        from .vocabulary import Vocabulary

        network, checkpoint = load_checkpoint(Path(checkpoint_path))
        vocabulary = Vocabulary(checkpoint["vocabulary"], checkpoint["hyperparams"]["char_ngrams"])
        self._model = TrainedModel(network, vocabulary)
        self._predict_proba = predict_proba
        self.labels = checkpoint["labels"]

    def predict(self, text: str) -> Prediction:
        probs = self._predict_proba(self._model, [text])[0]
        ranked = _rank_labels(self.labels, probs)
        return Prediction(ranked, self._model.vocabulary.known_word_fraction(text))


def _rank_labels(labels: list[str], probs) -> list[tuple[str, float]]:
    """Pair every label with its probability, most likely first."""
    most_likely_first = probs.argsort()[::-1]
    ranked = []
    for i in most_likely_first:
        ranked.append((labels[i], float(probs[i])))
    return ranked


class GeminiClassifier(Classifier):
    """Placeholder for a Gemini Flash-Lite backend. Not implemented yet.

    To implement it:
      1. pip install google-genai, and set the GEMINI_API_KEY environment variable.
      2. In predict(), send a prompt that lists self.labels and the user's text, and asks for
         exactly one label back (structured JSON output is the most reliable).
      3. Check the answer is in self.labels and return Prediction([(label, 1.0)]).
         known_words stays None; the chatbot then skips its "not a first-aid question" check.
    Nothing else needs to change: chat.py --backend gemini will use it.
    """

    def __init__(self, labels: list[str]):
        self.labels = labels
        raise NotImplementedError("The Gemini backend is a placeholder. See GeminiClassifier in firstaid/backends.py.")

    def predict(self, text: str) -> Prediction:
        raise NotImplementedError
