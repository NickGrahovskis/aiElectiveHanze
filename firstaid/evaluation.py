"""Step 4: feedback. How well did the trained model do, and what should be improved?"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import Guidance


@dataclass(frozen=True)
class LabelScore:
    label: str
    precision: float    # when the model says this label, how often it is right
    recall: float       # of the sentences that really have this label, how many it found
    f1: float           # precision and recall combined into one number
    sentences: int      # how many test sentences have this label


@dataclass(frozen=True)
class Mistake:
    text: str
    truth: Guidance
    guess: Guidance
    confidence: float   # how sure the model was of its wrong guess

    @property
    def is_under_triage(self) -> bool:
        """The guess is LESS urgent than the truth: the dangerous kind of mistake."""
        return self.guess.urgency_rank < self.truth.urgency_rank


def feedback_report(probabilities: np.ndarray, true_labels: np.ndarray, texts: list[str], labels: list[str],
                    guidance: dict[str, Guidance], train_accuracy: float, train_size: int,
                    cv_accuracy: float | None) -> str:
    """Human-readable report on the test set: accuracy, per-label scores, confusions, mistakes, advice."""
    predicted_labels = probabilities.argmax(axis=1)
    test_accuracy = float((predicted_labels == true_labels).mean())
    scores = _score_each_label(true_labels, predicted_labels, labels)
    mistakes = _find_mistakes(probabilities, true_labels, texts, labels, guidance)
    test_size = len(true_labels)
    test_correct = test_size - len(mistakes)
    advice = _advice(mistakes, scores, train_accuracy, test_accuracy, cv_accuracy, test_size)
    width = _longest(labels)  # label column width, so the tables line up

    sections = [
        _accuracy_section(train_accuracy, train_size, cv_accuracy, test_accuracy, test_correct, test_size),
        _per_label_section(scores, width),
        _confusion_section(true_labels, predicted_labels, labels, width),
        _mistakes_section(mistakes),
        _advice_section(advice),
    ]
    return _join_sections(sections)


# ---- working out the numbers ----

def _score_each_label(true_labels: np.ndarray, predicted_labels: np.ndarray, labels: list[str]) -> list[LabelScore]:
    label_ids = list(range(len(labels)))
    precisions, recalls, f1_scores, sentence_counts = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=label_ids, zero_division=0)
    scores = []
    for label_number, label in enumerate(labels):
        scores.append(LabelScore(label, float(precisions[label_number]), float(recalls[label_number]),
                                 float(f1_scores[label_number]), int(sentence_counts[label_number])))
    return scores


def _find_mistakes(probabilities: np.ndarray, true_labels: np.ndarray, texts: list[str], labels: list[str],
                   guidance: dict[str, Guidance]) -> list[Mistake]:
    predicted_labels = probabilities.argmax(axis=1)
    mistakes = []
    for sentence_number in range(len(true_labels)):
        true_label = true_labels[sentence_number]
        predicted_label = predicted_labels[sentence_number]
        if predicted_label != true_label:
            truth = guidance[labels[true_label]]
            guess = guidance[labels[predicted_label]]
            confidence = float(probabilities[sentence_number, predicted_label])
            mistakes.append(Mistake(texts[sentence_number], truth, guess, confidence))
    return mistakes


def _count_under_triage(mistakes: list[Mistake]) -> int:
    count = 0
    for mistake in mistakes:
        if mistake.is_under_triage:
            count += 1
    return count


def _weak_labels(scores: list[LabelScore]) -> list[str]:
    """Labels with test sentences of which the model found fewer than half."""
    weak = []
    for score in scores:
        if score.sentences and score.recall < 0.5:
            weak.append(score.label)
    return weak


def _advice(mistakes: list[Mistake], scores: list[LabelScore], train_accuracy: float, test_accuracy: float,
            cv_accuracy: float | None, test_size: int) -> list[str]:
    """Tips based on the numbers. The last tip is always a note on the test set's size."""
    advice = []
    under_triage = _count_under_triage(mistakes)
    if under_triage:
        advice.append(f"{under_triage} test sentence(s) were predicted as LESS urgent than they are. "
                      "This is the dangerous kind of mistake: add more example sentences for those labels.")
    if train_accuracy - test_accuracy >= 0.15:
        advice.append("The model does much better on training data than on new data (overfitting). "
                      "Add more varied sentences per label, or search higher dropout / weight_decay.")
    if train_accuracy < 0.9:
        advice.append("The model does not even fit the training data well (underfitting). "
                      "Try more epochs, a bigger hidden layer or a higher learning rate.")
    weak = _weak_labels(scores)
    if weak:
        advice.append(f"Weak labels (recall under 50%): {', '.join(weak)}. Add more varied sentences for them in Sheet1.")
    if cv_accuracy is not None and abs(cv_accuracy - test_accuracy) > 0.15:
        advice.append("Cross-validation and test accuracy disagree a lot; with so few test sentences this can be luck. "
                      "Try another --seed to see how stable the result is.")
    if not advice:
        advice.append("Looks good. Try the chatbot with your own wording; any sentence it gets wrong is a great new "
                      "training example for Sheet1.")
    advice.append(f"Note: the test set has {test_size} sentences, so one mistake moves test accuracy by {1 / test_size:.1%}.")
    return advice


# ---- writing the report, one section at a time ----

def _accuracy_section(train_accuracy: float, train_size: int, cv_accuracy: float | None,
                      test_accuracy: float, test_correct: int, test_size: int) -> list[str]:
    lines = ["Accuracy",
             f"  training data      {train_accuracy:6.1%}   ({train_size} sentences the model learned from)"]
    if cv_accuracy is not None:
        lines.append(f"  cross-validation   {cv_accuracy:6.1%}   (estimate from tuning, on the training split)")
    lines.append(f"  test data          {test_accuracy:6.1%}   ({test_correct}/{test_size} sentences the model never saw)")
    return lines


def _per_label_section(scores: list[LabelScore], width: int) -> list[str]:
    lines = ["Per label (test set)",
             f"  {'label':<{width}}  precision  recall     f1  sentences"]
    for score in scores:
        lines.append(f"  {score.label:<{width}}  {score.precision:9.0%}  {score.recall:6.0%}  "
                     f"{score.f1:5.0%}  {score.sentences:9d}")
    return lines


def _confusion_section(true_labels: np.ndarray, predicted_labels: np.ndarray, labels: list[str],
                       width: int) -> list[str]:
    """A table: row = true label, column = predicted label. The diagonal holds the correct answers."""
    label_ids = list(range(len(labels)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_ids)
    lines = ["Confusion matrix (rows = true label, columns = predicted label number)",
             " " * (width + 7) + _column_numbers(len(labels))]
    for label_number, label in enumerate(labels):
        lines.append(f"  {label_number:>2} {label:<{width}}  {_matrix_row(matrix[label_number])}")
    return lines


def _column_numbers(count: int) -> str:
    """'  0  1  2 ...'"""
    header = ""
    for column_number in range(count):
        header += f"{column_number:>3}"
    return header


def _matrix_row(counts: np.ndarray) -> str:
    """Each count in 3 characters. Zeros are shown as '.', so the confusions stand out."""
    row = ""
    for count in counts:
        if count:
            row += f"{count:>3}"
        else:
            row += "  ."
    return row


def _mistakes_section(mistakes: list[Mistake]) -> list[str]:
    lines = [f"Mistakes on the test set ({len(mistakes)}, of which {_count_under_triage(mistakes)} under-triage = "
             "predicted LESS urgent than reality)"]
    if not mistakes:
        lines.append("  none")
    for mistake in mistakes:
        lines += _describe_mistake(mistake)
    return lines


def _describe_mistake(mistake: Mistake) -> list[str]:
    flag = ""
    if mistake.is_under_triage:
        flag = "  [UNDER-TRIAGE]"
    truth = mistake.truth
    guess = mistake.guess
    return [f'  "{mistake.text}"{flag}',
            f"      true: {truth.label} ({truth.urgency})   "
            f"predicted: {guess.label} ({guess.urgency}, {mistake.confidence:.0%} sure)"]


def _advice_section(advice: list[str]) -> list[str]:
    lines = ["Advice"]
    for tip in advice:
        lines.append(f"  - {tip}")
    return lines


def _join_sections(sections: list[list[str]]) -> str:
    """Join the sections into one text, with an empty line between them."""
    texts = []
    for section in sections:
        texts.append("\n".join(section))
    return "\n\n".join(texts)


def _longest(texts: list[str]) -> int:
    """Length of the longest text."""
    longest = 0
    for text in texts:
        longest = max(longest, len(text))
    return longest
