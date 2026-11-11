from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from .model import FirstAidNet, pack
from .mpnet import SentenceEncoder, load_mpnet
from .vocabulary import Vocabulary


@dataclass(frozen=True)
class HyperParams:
    embedding_dim: int = 64         # numbers per learned embedding vector
    char_ngrams: bool = True        # also use 3-5 letter word pieces as features
    mpnet: bool = False             # use mpnet's fixed sentence vectors instead of learned embeddings
    hidden_size: int = 64           # neurons per hidden layer
    num_hidden_layers: int = 1      # how many hidden layers
    dropout: float = 0.2            # fraction of neurons switched off during training (fights overfitting)
    learning_rate: float = 0.01     # how big each weight/bias adjustment is
    weight_decay: float = 1e-4      # pulls weights towards zero (fights overfitting)
    epochs: int = 100               # passes over the training data
    batch_size: int = 8             # sentences per weight update

    def short(self) -> str:
        return (f"{self._short_embeddings()} "
                f"hidden={self.hidden_size}x{self.num_hidden_layers} drop={self.dropout} "
                f"lr={self.learning_rate:g} wd={self.weight_decay:g} ep={self.epochs} bs={self.batch_size}")

    def _short_embeddings(self) -> str:
        if self.mpnet:
            return "emb=mpnet"
        pieces = "off"
        if self.char_ngrams:
            pieces = "on"
        return f"emb={self.embedding_dim} pieces={pieces}"


# The values random_search() picks from. Edit these to widen or narrow the search.
# With mpnet the first two are not used (mpnet provides the embeddings), so they are skipped.
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
NOT_USED_WITH_MPNET = ("embedding_dim", "char_ngrams")


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
    mpnet: SentenceEncoder | None = None    # None when the embeddings were learned from scratch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---- training: adjusting weights and biases ----

def fit(texts: list[str], true_labels: np.ndarray, hyperparams: HyperParams, num_classes: int, seed: int = 42,
        log_every: int = 0) -> TrainedModel:
    """Build a vocabulary from `texts` and train a fresh network on them.

    With log_every > 0, print progress every N epochs.
    """
    set_seed(seed)

    # The inputs: feature IDs per sentence, plus each sentence's mpnet vector when mpnet is used.
    vocabulary = Vocabulary.build(texts, hyperparams.char_ngrams)
    encoded_sentences = vocabulary.encode_all(texts)
    mpnet = _mpnet_if_used(hyperparams)
    mpnet_vectors = _mpnet_vectors(mpnet, texts)
    true_labels_tensor = torch.from_numpy(true_labels).long()

    # The network, and how it learns.
    network = _new_network(len(vocabulary), num_classes, hyperparams)
    model = TrainedModel(network, vocabulary, mpnet)
    optimizer = torch.optim.AdamW(network.parameters(), lr=hyperparams.learning_rate,
                                  weight_decay=hyperparams.weight_decay)
    loss_function = nn.CrossEntropyLoss()
    shuffle_generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, hyperparams.epochs + 1):
        epoch_loss = _train_one_epoch(network, encoded_sentences, mpnet_vectors, true_labels_tensor,
                                      optimizer, loss_function, hyperparams.batch_size, shuffle_generator)
        if _should_log(epoch, hyperparams.epochs, log_every):
            train_accuracy = accuracy(predict_proba(model, texts), true_labels)
            print(f"    epoch {epoch:4d}/{hyperparams.epochs}   loss {epoch_loss:.4f}   "
                  f"train accuracy {train_accuracy:.1%}")
    network.eval()
    return model


def _new_network(vocab_size: int, num_classes: int, hyperparams: HyperParams) -> FirstAidNet:
    return FirstAidNet(vocab_size, num_classes, hyperparams.embedding_dim, hyperparams.hidden_size,
                       hyperparams.num_hidden_layers, hyperparams.dropout, hyperparams.mpnet)


def _mpnet_if_used(hyperparams: HyperParams) -> SentenceEncoder | None:
    if not hyperparams.mpnet:
        return None
    return load_mpnet()


def _mpnet_vectors(mpnet: SentenceEncoder | None, texts: list[str]) -> torch.Tensor | None:
    """Each sentence's mpnet vector (one row per sentence), or None without mpnet."""
    if mpnet is None:
        return None
    return torch.tensor(mpnet.sentence_vectors(texts), dtype=torch.float32)


def _train_one_epoch(network: FirstAidNet, encoded_sentences: list[list[int]], mpnet_vectors: torch.Tensor | None,
                     true_labels_tensor: torch.Tensor, optimizer: torch.optim.Optimizer, loss_function: nn.Module,
                     batch_size: int, shuffle_generator: torch.Generator) -> float:
    """One pass over all training sentences, in a new random order. Returns the average loss."""
    network.train()
    shuffled_positions = torch.randperm(len(encoded_sentences), generator=shuffle_generator).tolist()
    total_loss = 0.0
    for batch_positions in _split_into_batches(shuffled_positions, batch_size):
        batch_sentences = _pick(encoded_sentences, batch_positions)
        batch_mpnet_vectors = _pick_rows(mpnet_vectors, batch_positions)
        batch_true_labels = true_labels_tensor[batch_positions]
        batch_loss = _train_one_batch(network, batch_sentences, batch_mpnet_vectors, batch_true_labels,
                                      optimizer, loss_function)
        total_loss += batch_loss * len(batch_positions)
    return total_loss / len(encoded_sentences)


def _train_one_batch(network: FirstAidNet, batch_sentences: list[list[int]],
                     batch_mpnet_vectors: torch.Tensor | None, batch_true_labels: torch.Tensor,
                     optimizer: torch.optim.Optimizer, loss_function: nn.Module) -> float:
    """One adjustment of all weights and biases. Returns the batch's loss."""
    token_ids, offsets = pack(batch_sentences)
    label_scores = network(token_ids, offsets, batch_mpnet_vectors)  # forward pass: scores per label
    loss = loss_function(label_scores, batch_true_labels)            # how wrong were we?
    optimizer.zero_grad()                                            # forget the previous batch's gradients
    loss.backward()                                                  # gradient for every weight and bias
    optimizer.step()                                                 # adjust weights and biases
    return loss.item()


def _split_into_batches(shuffled_positions: list[int], batch_size: int) -> list[list[int]]:
    """[5, 2, 8, 1, 0] with batch_size 2 -> [[5, 2], [8, 1], [0]]"""
    batches = []
    for start in range(0, len(shuffled_positions), batch_size):
        batches.append(shuffled_positions[start:start + batch_size])
    return batches


def _pick(items: list, positions) -> list:
    """The items at the given positions: _pick(['a', 'b', 'c'], [2, 0]) -> ['c', 'a']"""
    picked = []
    for position in positions:
        picked.append(items[position])
    return picked


def _pick_rows(vectors: torch.Tensor | None, positions: list[int]) -> torch.Tensor | None:
    """The rows at the given positions, or None when there are no vectors (no mpnet)."""
    if vectors is None:
        return None
    return vectors[positions]


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
    token_ids, offsets = pack(model.vocabulary.encode_all(texts))
    mpnet_vectors = _mpnet_vectors(model.mpnet, texts)
    label_scores = model.network(token_ids, offsets, mpnet_vectors)
    return torch.softmax(label_scores, dim=1).numpy()


def accuracy(probabilities: np.ndarray, true_labels: np.ndarray) -> float:
    """Share of sentences whose most likely label is the true label."""
    return float((probabilities.argmax(axis=1) == true_labels).mean())


def cross_entropy(probabilities: np.ndarray, true_labels: np.ndarray) -> float:
    """Average of -log(probability given to the true label). Lower = more confident when right."""
    true_label_probs = probabilities[np.arange(len(true_labels)), true_labels]
    return float(-np.log(true_label_probs + 1e-12).mean())


# ---- tuning: choosing hyperparameters ----

def random_search(texts: list[str], true_labels: np.ndarray, num_classes: int, trials: int = 40, folds: int = 4,
                  seed: int = 42, on_trial: Callable[[int, int, TrialResult], None] | None = None,
                  base: HyperParams = HyperParams()) -> list[TrialResult]:
    """Try `trials` hyperparameter combinations; return them best first.

    `base` gives the values of hyperparameters that are not in SEARCH_SPACE (such as mpnet).
    After each trial, on_trial(trial number, number of trials, result) is called if given.
    """
    candidates = _pick_candidates(trials, seed, base)
    results = []
    for number, hyperparams in enumerate(candidates, start=1):
        result = cross_validate(texts, true_labels, hyperparams, num_classes, folds, seed)
        results.append(result)
        if on_trial:
            on_trial(number, len(candidates), result)
    return sorted(results, key=_ranking)


def _pick_candidates(trials: int, seed: int, base: HyperParams) -> list[HyperParams]:
    """`base` first (as a baseline), then random combinations from SEARCH_SPACE, no repeats."""
    random_generator = random.Random(seed)
    wanted = min(trials, _count_combinations(base))
    candidates = [base]
    while len(candidates) < wanted:
        hyperparams = _random_hyperparams(random_generator, base)
        if hyperparams not in candidates:
            candidates.append(hyperparams)
    return candidates


def _count_combinations(base: HyperParams) -> int:
    """How many different combinations SEARCH_SPACE allows (3 x 2 x 3 x ...)."""
    total = 1
    for name in _searched(base):
        total *= len(SEARCH_SPACE[name])
    return total


def _random_hyperparams(random_generator: random.Random, base: HyperParams) -> HyperParams:
    """A random value from SEARCH_SPACE for every searched hyperparameter; the rest come from `base`."""
    chosen = {}
    for name in _searched(base):
        chosen[name] = random_generator.choice(SEARCH_SPACE[name])
    return replace(base, **chosen)


def _searched(base: HyperParams) -> list[str]:
    """The SEARCH_SPACE names worth trying: all of them, minus the embedding ones with mpnet."""
    names = []
    for name in SEARCH_SPACE:
        if base.mpnet and name in NOT_USED_WITH_MPNET:
            continue
        names.append(name)
    return names


def _ranking(result: TrialResult) -> tuple[float, float]:
    """Sort key: highest cross-validation accuracy first; ties go to the lowest loss."""
    return -result.cv_accuracy, result.cv_loss


def cross_validate(texts: list[str], true_labels: np.ndarray, hyperparams: HyperParams, num_classes: int,
                   folds: int = 4, seed: int = 42) -> TrialResult:
    """Train on k-1 folds, validate on the remaining one, k times. Every label appears in every fold.

    The vocabulary is rebuilt from the training folds each time, so validation words stay unseen.
    """
    accuracies = []
    losses = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    placeholder = np.zeros(len(true_labels))  # split() wants inputs too, but only looks at true_labels
    for train_positions, validation_positions in splitter.split(placeholder, true_labels):
        fold_accuracy, fold_loss = _score_fold(texts, true_labels, train_positions, validation_positions,
                                               hyperparams, num_classes, seed)
        accuracies.append(fold_accuracy)
        losses.append(fold_loss)
    return TrialResult(hyperparams, float(np.mean(accuracies)), float(np.std(accuracies)), float(np.mean(losses)))


def _score_fold(texts: list[str], true_labels: np.ndarray, train_positions: np.ndarray,
                validation_positions: np.ndarray, hyperparams: HyperParams, num_classes: int,
                seed: int) -> tuple[float, float]:
    """Train on one fold's training part; return (accuracy, loss) on its validation part."""
    model = fit(_pick(texts, train_positions), true_labels[train_positions], hyperparams, num_classes, seed)
    probabilities = predict_proba(model, _pick(texts, validation_positions))
    validation_true_labels = true_labels[validation_positions]
    return accuracy(probabilities, validation_true_labels), cross_entropy(probabilities, validation_true_labels)
