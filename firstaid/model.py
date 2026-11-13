"""Step 2: the neural network, trained completely from scratch.
        -> Linear (weights W1, biases b1) -> ReLU -> Dropout     } repeated num_hidden_layers times
        -> Linear (weights W2, biases b2)
        -> one score (logit) per label; softmax turns them into probabilities

The embedding table is a weight matrix too (one row per vocabulary feature). Every nn.Linear layer
computes  output = input @ W.T + b.  Training (see training.py) adjusts all of them.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from . import mpnet


class FirstAidNet(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 64, hidden_size: int = 64,
                 num_hidden_layers: int = 1, dropout: float = 0.2, use_mpnet: bool = False,
                 use_own_embeddings: bool = False):
        super().__init__()
        self.use_mpnet = use_mpnet
        self.embedding = None
        input_size = 0
        if use_own_embeddings or not use_mpnet:
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
            input_size += embedding_dim
        if use_mpnet:
            # Fixed meaning from mpnet: copes with wordings never seen in training.
            input_size += mpnet.DIM

        self.layers = _build_layers(input_size, hidden_size, num_hidden_layers, dropout, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Create the starting weights and biases.

        Embedding vectors and layer weights start as small random numbers (so every feature and
        neuron starts out different), biases start at zero. Training then adjusts all of them.
        """
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def embed(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Sentence embeddings: the average of the learned vectors of each sentence's features."""
        return self.embedding(token_ids, offsets)

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor,
                mpnet_vectors: torch.Tensor | None = None) -> torch.Tensor:
        parts = []
        if self.embedding is not None:
            parts.append(self.embed(token_ids, offsets))
        if self.use_mpnet:
            parts.append(mpnet_vectors)
        return self.layers(torch.cat(parts, dim=1))

    def count_parameters(self) -> tuple[int, int]:
        """(number of weights, number of biases). Embedding vectors count as weights."""
        weights = 0
        biases = 0
        for name, parameter in self.named_parameters():
            if name.endswith("weight"):
                weights += parameter.numel()
            elif name.endswith("bias"):
                biases += parameter.numel()
        return weights, biases

def _build_layers(input_size: int, hidden_size: int, num_hidden_layers: int, dropout: float,
                  num_classes: int) -> nn.Sequential:
    """The hidden layers, then an output layer with one score per label."""
    layers: list[nn.Module] = []
    in_features = input_size
    for layer_number in range(num_hidden_layers):
        layers += _hidden_layer(in_features, hidden_size, dropout)
        in_features = hidden_size
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


def _hidden_layer(in_features: int, hidden_size: int, dropout: float) -> list[nn.Module]:
    return [nn.Linear(in_features, hidden_size), nn.ReLU(), nn.Dropout(dropout)]


def pack(encoded_sentences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a batch of ID lists into the (flat IDs, start offsets) pair that EmbeddingBag expects.

    [[4, 7], [2, 9, 5]] -> flat IDs [4, 7, 2, 9, 5], offsets [0, 2] (where each sentence starts)
    """
    all_feature_ids = []
    offsets = []
    for sentence_feature_ids in encoded_sentences:
        offsets.append(len(all_feature_ids))
        all_feature_ids.extend(sentence_feature_ids)
    return torch.tensor(all_feature_ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def save_trained_model(path: Path, model: FirstAidNet, hyperparams: dict, labels: list[str],
                    vocabulary_items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyperparams": hyperparams,
        "labels": labels,
        "vocabulary": vocabulary_items,
    }, path)


def load_previous_model(path: Path) -> tuple[FirstAidNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hyperparams = checkpoint["hyperparams"]
    model = FirstAidNet(
        vocab_size=len(checkpoint["vocabulary"]),
        num_classes=len(checkpoint["labels"]),
        embedding_dim=hyperparams["embedding_dim"],
        hidden_size=hyperparams["hidden_size"],
        num_hidden_layers=hyperparams["num_hidden_layers"],
        dropout=hyperparams["dropout"],
        use_mpnet=hyperparams.get("mpnet", False),  # models saved before --mpnet existed have no "mpnet" entry
        use_own_embeddings=hyperparams.get("own_embeddings", False),  # older mpnet models used mpnet only
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
