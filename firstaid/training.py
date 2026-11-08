"""Step 3: hyperparameters, training (adjusting weights and biases) and automatic tuning.

Weights and biases are *learned*: each training step runs the network, measures how wrong it was
(cross-entropy loss), computes the gradient of that loss for every weight and bias (backpropagation),
and nudges them in the direction that lowers the loss (AdamW optimiser). This includes the
embedding vectors, so the model learns its own embeddings from your sentences.

Hyperparameters are *chosen*: embedding size, layer size, learning rate, etc. random_search() tries
many combinations and scores each with k-fold cross-validation on the training split, so the test
split stays unseen until the final check.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from .model import FirstAidNet, pack
from .vocabulary import Vocabulary


@dataclass(frozen=True)
class HyperParams:
    embedding_dim: int = 64         # numbers per learned embedding vector
    char_ngrams: bool = True        # also use 3-5 letter word pieces as features
    hidden_size: int = 64           # neurons per hidden layer
    num_hidden_layers: int = 1      # how many hidden layers
    dropout: float = 0.2            # fraction of neurons switched off during training (fights overfitting)
    learning_rate: float = 0.01     # how big each weight/bias adjustment is
    weight_decay: float = 1e-4      # pulls weights towards zero (fights overfitting)
    epochs: int = 100               # passes over the training data
    batch_size: int = 8             # sentences per weight update

    def short(self) -> str:
        pieces = "off"
        if self.char_ngrams:
            pieces = "on"
        return (f"emb={self.embedding_dim} pieces={pieces} "
                f"hidden={self.hidden_size}x{self.num_hidden_layers} drop={self.dropout} "
                f"lr={self.learning_rate:g} wd={self.weight_decay:g} ep={self.epochs} bs={self.batch_size}")


# Values random_search() picks from. Edit these to widen or narrow the search.
SEARCH_SPACE: dict[str, list] = {
    "embedding_dim": [32, 64, 128],
    "char_ngrams": [True, False],
    "hidden_size": [32, 64, 128],
    "num_hidden_layers": [1, 2],
    "dropout": [0.0, 0.2, 0.4],
    "learning_rate": [0.003, 0.01, 0.03],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "epochs": [50, 100, 200],
    "batch_size": [8, 16],
}


@dataclass(frozen=True)
class TrialResult:
    hyperparams: HyperParams
    cv_accuracy: float      # mean validation accuracy over the folds
    cv_accuracy_std: float
    cv_loss: float          # mean validation cross-entropy (lower = more confident when right)

    def to_dict(self) -> dict:
        return {"hyperparams": asdict(self.hyperparams), "cv_accuracy": self.cv_accuracy,
                "cv_accuracy_std": self.cv_accuracy_std, "cv_loss": self.cv_loss}


@dataclass
class TrainedModel:
    network: FirstAidNet
    vocabulary: Vocabulary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---- training: adjusting weights and biases ----

def fit(texts: list[str], y: np.ndarray, hp: HyperParams, num_classes: int, seed: int = 42,
        log_every: int = 0) -> TrainedModel:
    """Build a vocabulary from `texts` and train a fresh network on them.

    With log_every > 0, print progress every N epochs.
    """
    set_seed(seed)
    vocabulary = Vocabulary.build(texts, hp.char_ngrams)
    encoded = vocabulary.encode_all(texts)
    network = _new_network(len(vocabulary), num_classes, hp)
    model = TrainedModel(network, vocabulary)
    optimizer = torch.optim.AdamW(network.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    targets = torch.from_numpy(y).long()
    shuffle = torch.Generator().manual_seed(seed)

    for epoch in range(1, hp.epochs + 1):
        epoch_loss = _train_one_epoch(network, encoded, targets, optimizer, loss_fn, hp.batch_size, shuffle)
        if _should_log(epoch, hp.epochs, log_every):
            train_accuracy = accuracy(predict_proba(model, texts), y)
            print(f"    epoch {epoch:4d}/{hp.epochs}   loss {epoch_loss:.4f}   train accuracy {train_accuracy:.1%}")
    network.eval()
    return model


def _new_network(vocab_size: int, num_classes: int, hp: HyperParams) -> FirstAidNet:
    return FirstAidNet(vocab_size, num_classes, hp.embedding_dim, hp.hidden_size,
                       hp.num_hidden_layers, hp.dropout)


def _train_one_epoch(network: FirstAidNet, encoded: list[list[int]], targets: torch.Tensor,
                     optimizer: torch.optim.Optimizer, loss_fn: nn.Module, batch_size: int,
                     shuffle: torch.Generator) -> float:
    """One pass over all training sentences, in a new random order. Returns the average loss."""
    network.train()
    order = torch.randperm(len(encoded), generator=shuffle).tolist()
    total_loss = 0.0
    for batch in _split_into_batches(order, batch_size):
        batch_loss = _train_one_batch(network, _pick(encoded, batch), targets[batch], optimizer, loss_fn)
        total_loss += batch_loss * len(batch)
    return total_loss / len(encoded)


def _train_one_batch(network: FirstAidNet, batch_encoded: list[list[int]], batch_targets: torch.Tensor,
                     optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    """One adjustment of all weights and biases. Returns the batch's loss."""
    logits = network(*pack(batch_encoded))   # forward pass: scores per label
    loss = loss_fn(logits, batch_targets)    # how wrong were we?
    optimizer.zero_grad()
    loss.backward()                          # gradient for every weight and bias
    optimizer.step()                         # adjust weights and biases
    return loss.item()


def _split_into_batches(order: list[int], batch_size: int) -> list[list[int]]:
    """[5, 2, 8, 1, 0] with batch_size 2 -> [[5, 2], [8, 1], [0]]"""
    batches = []
    for start in range(0, len(order), batch_size):
        batches.append(order[start:start + batch_size])
    return batches


def _pick(items: list, positions) -> list:
    """The items at the given positions: _pick(['a', 'b', 'c'], [2, 0]) -> ['c', 'a']"""
    picked = []
    for position in positions:
        picked.append(items[position])
    return picked


def _should_log(epoch: int, total_epochs: int, log_every: int) -> bool:
    """Every `log_every` epochs, and after the last one. log_every = 0 means never."""
    if log_every == 0:
        return False
    return epoch % log_every == 0 or epoch == total_epochs


# ---- using a trained model ----

@torch.no_grad()
def predict_proba(model: TrainedModel, texts: list[str]) -> np.ndarray:
    """The probability of every label for every sentence: an array of shape (sentences, labels)."""
    model.network.eval()
    logits = model.network(*pack(model.vocabulary.encode_all(texts)))
    return torch.softmax(logits, dim=1).numpy()


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    """Share of sentences whose most likely label is the true label."""
    return float((probs.argmax(axis=1) == y).mean())


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Average of -log(probability given to the true label). Lower = more confident when right."""
    true_label_probs = probs[np.arange(len(y)), y]
    return float(-np.log(true_label_probs + 1e-12).mean())


# ---- tuning: choosing hyperparameters ----

def random_search(texts: list[str], y: np.ndarray, num_classes: int, trials: int = 40, folds: int = 4,
                  seed: int = 42, on_trial: Callable[[int, int, TrialResult], None] | None = None) -> list[TrialResult]:
    """Try `trials` hyperparameter combinations; return them best first.

    After each trial, on_trial(trial number, number of trials, result) is called if given.
    """
    candidates = _pick_candidates(trials, seed)
    results = []
    for number, hp in enumerate(candidates, start=1):
        result = cross_validate(texts, y, hp, num_classes, folds, seed)
        results.append(result)
        if on_trial:
            on_trial(number, len(candidates), result)
    return sorted(results, key=_ranking)


def _pick_candidates(trials: int, seed: int) -> list[HyperParams]:
    """The defaults first (as a baseline), then random combinations from SEARCH_SPACE, no repeats."""
    rng = random.Random(seed)
    wanted = min(trials, _count_combinations())
    candidates = [HyperParams()]
    while len(candidates) < wanted:
        hp = _random_hyperparams(rng)
        if hp not in candidates:
            candidates.append(hp)
    return candidates


def _count_combinations() -> int:
    """How many different combinations SEARCH_SPACE allows (3 x 2 x 3 x ...)."""
    total = 1
    for values in SEARCH_SPACE.values():
        total *= len(values)
    return total


def _random_hyperparams(rng: random.Random) -> HyperParams:
    """A random value from SEARCH_SPACE for every hyperparameter."""
    chosen = {}
    for name, values in SEARCH_SPACE.items():
        chosen[name] = rng.choice(values)
    return HyperParams(**chosen)


def _ranking(result: TrialResult) -> tuple[float, float]:
    """Sort key: highest cross-validation accuracy first; ties go to the lowest loss."""
    return -result.cv_accuracy, result.cv_loss


def cross_validate(texts: list[str], y: np.ndarray, hp: HyperParams, num_classes: int,
                   folds: int = 4, seed: int = 42) -> TrialResult:
    """Train on k-1 folds, validate on the remaining one, k times. Every label appears in every fold.

    The vocabulary is rebuilt from the training folds each time, so validation words stay unseen.
    """
    accuracies = []
    losses = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(np.zeros(len(y)), y):
        fold_accuracy, fold_loss = _score_fold(texts, y, train_idx, val_idx, hp, num_classes, seed)
        accuracies.append(fold_accuracy)
        losses.append(fold_loss)
    return TrialResult(hp, float(np.mean(accuracies)), float(np.std(accuracies)), float(np.mean(losses)))


def _score_fold(texts: list[str], y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                hp: HyperParams, num_classes: int, seed: int) -> tuple[float, float]:
    """Train on one fold's training part; return (accuracy, loss) on its validation part."""
    model = fit(_pick(texts, train_idx), y[train_idx], hp, num_classes, seed)
    probs = predict_proba(model, _pick(texts, val_idx))
    return accuracy(probs, y[val_idx]), cross_entropy(probs, y[val_idx])
