"""Train the first-aid classifier from scratch and print feedback.

    python train.py                 # tune hyperparameters (40 trials), train, evaluate, save
    python train.py --trials 80     # search harder
    python train.py --no-tune       # skip tuning, use the default HyperParams
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from firstaid.data import DEFAULT_WORKBOOK, PROJECT_DIR, Dataset, Scenario, load_dataset
from firstaid.evaluation import feedback_report
from firstaid.model import save_checkpoint
from firstaid.training import HyperParams, TrainedModel, TrialResult, accuracy, fit, predict_proba, random_search


@dataclass(frozen=True)
class Sentences:
    """The sentences of one split, and the true label of each as a number (its position in the label list)."""
    texts: list[str]
    targets: np.ndarray


def main() -> None:
    args = parse_args()
    started = time.time()

    dataset, train, test = load_data(args.data)                               # 1)
    best, trials = choose_hyperparams(train, len(dataset.labels), args)       # 2)
    model = train_final_model(train, best, len(dataset.labels), args.seed)    # 3)
    report = give_feedback(model, train, test, dataset, trials)               # 4)
    save_results(args.out, model, best, dataset.labels, report, trials, args.seed)

    print(f"\nSaved model and report to {args.out}  ({time.time() - started:.0f}s)")
    print("Chat with it:  python chat.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_WORKBOOK, help="the scenarios workbook")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "artifacts", help="where to save the model and report")
    parser.add_argument("--trials", type=int, default=40, help="hyperparameter combinations to try")
    parser.add_argument("--folds", type=int, default=4, help="cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="random seed (same seed = same result)")
    parser.add_argument("--no-tune", action="store_true", help="skip tuning and use the default hyperparameters")
    return parser.parse_args()


# ---- 1) loading data ----

def load_data(path: Path) -> tuple[Dataset, Sentences, Sentences]:
    """The whole workbook, plus its train and test sentences."""
    print("1) Loading data")
    dataset = load_dataset(path)
    train = sentences_of(dataset.split("train"), dataset.labels)
    test = sentences_of(dataset.split("test"), dataset.labels)
    print(f"   {len(dataset.scenarios)} scenarios, {len(dataset.labels)} labels: "
          f"{len(train.texts)} train / {len(test.texts)} test")
    return dataset, train, test


def sentences_of(scenarios: list[Scenario], labels: list[str]) -> Sentences:
    texts = []
    targets = []
    for scenario in scenarios:
        texts.append(scenario.text)
        targets.append(labels.index(scenario.label))
    return Sentences(texts, np.array(targets))


# ---- 2) tuning hyperparameters ----

def choose_hyperparams(train: Sentences, num_labels: int,
                       args: argparse.Namespace) -> tuple[HyperParams, list[TrialResult]]:
    """The best hyperparameters, and every trial tried (best first). No trials with --no-tune."""
    if args.no_tune:
        print("\n2) Tuning skipped (--no-tune), using the default hyperparameters")
        return HyperParams(), []

    print(f"\n2) Tuning hyperparameters: {args.trials} combinations x {args.folds}-fold cross-validation")
    trials = random_search(train.texts, train.targets, num_labels, args.trials, args.folds, args.seed,
                           on_trial=print_trial)
    print_top_trials(trials, 5)
    return trials[0].hyperparams, trials


def print_trial(number: int, total: int, result: TrialResult) -> None:
    print(f"   trial {number:3d}/{total}   cv accuracy {result.cv_accuracy:6.1%} "
          f"+/- {result.cv_accuracy_std:4.1%}   {result.hyperparams.short()}")


def print_top_trials(trials: list[TrialResult], count: int) -> None:
    print(f"\n   Top {count}:")
    for rank, result in enumerate(trials[:count], start=1):
        print(f"   {rank}. cv accuracy {result.cv_accuracy:6.1%}  loss {result.cv_loss:.3f}   {result.hyperparams.short()}")


# ---- 3) training the final network ----

def train_final_model(train: Sentences, hp: HyperParams, num_labels: int, seed: int) -> TrainedModel:
    print(f"\n3) Training the final network from scratch on all {len(train.texts)} training sentences")
    print(f"   {hp.short()}")
    model = fit(train.texts, train.targets, hp, num_labels, seed, log_every=max(1, hp.epochs // 10))
    print_model_size(model, hp, num_labels)
    return model


def print_model_size(model: TrainedModel, hp: HyperParams, num_labels: int) -> None:
    weights, biases = model.network.count_parameters()
    print(f"   vocabulary: {len(model.vocabulary):,} features found in your sentences")
    print(f"   network: embeddings ({len(model.vocabulary):,} x {hp.embedding_dim}) -> "
          f"{describe_hidden_layers(hp)} -> {num_labels} labels")
    print(f"   learned: {weights:,} weights + {biases:,} biases")


def describe_hidden_layers(hp: HyperParams) -> str:
    """'hidden 128' for one hidden layer, 'hidden 64 -> hidden 64' for two."""
    layers = []
    for _ in range(hp.num_hidden_layers):
        layers.append(f"hidden {hp.hidden_size}")
    return " -> ".join(layers)


# ---- 4) feedback ----

def give_feedback(model: TrainedModel, train: Sentences, test: Sentences, dataset: Dataset,
                  trials: list[TrialResult]) -> str:
    """Print and return the report on the test sentences."""
    print("\n4) Feedback on the test set (sentences never used for training or tuning)\n")
    train_accuracy = accuracy(predict_proba(model, train.texts), train.targets)
    test_probs = predict_proba(model, test.texts)
    report = feedback_report(test_probs, test.targets, test.texts, dataset.labels, dataset.guidance,
                             train_accuracy, len(train.texts), best_cv_accuracy(trials))
    print(report)
    return report


def best_cv_accuracy(trials: list[TrialResult]) -> float | None:
    """The winning trial's cross-validation accuracy, or None when tuning was skipped."""
    if not trials:
        return None
    return trials[0].cv_accuracy


# ---- saving ----

def save_results(out: Path, model: TrainedModel, hp: HyperParams, labels: list[str], report: str,
                 trials: list[TrialResult], seed: int) -> None:
    """model.pt for the chatbot, plus the report and (after tuning) every trial, for you."""
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "model.pt", model.network, asdict(hp), labels, model.vocabulary.items)
    header = f"Hyperparameters: {hp.short()}\nSeed: {seed}\n\n"
    (out / "training_report.txt").write_text(header + report + "\n", encoding="utf-8")
    if trials:
        save_trials(out / "tuning_trials.json", trials)


def save_trials(path: Path, trials: list[TrialResult]) -> None:
    rows = []
    for trial in trials:
        rows.append(trial.to_dict())
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
