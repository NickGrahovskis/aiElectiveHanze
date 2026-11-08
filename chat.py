"""Terminal first-aid chatbot.

    python chat.py                                  # interactive chat
    python chat.py "I burned my hand on the oven"   # answer one message and exit
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from firstaid.backends import DEFAULT_CHECKPOINT, Classifier, GeminiClassifier, LocalClassifier, Prediction
from firstaid.data import DEFAULT_WORKBOOK, Guidance, load_dataset

DISCLAIMER = ("Educational prototype only. This is not medical advice and does not replace a doctor.\n"
              "In an emergency call 112.")
HELP = "Describe what happened in your own words, e.g. 'I cut my finger while cooking'. Type 'quit' to leave."
QUIT_WORDS = {"quit", "exit", "q"}
HELP_WORDS = {"help", "?"}
TOP_GUESSES = 3             # how many of the most likely labels the bot looks at
CAUTION_PROBABILITY = 0.15  # mention a more urgent runner-up label if it is at least this likely


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.data)
    classifier = load_classifier(args.backend, args.checkpoint, dataset.labels)

    if args.message:
        print_answer(args.message, classifier, dataset.guidance, args)
        return
    chat_loop(classifier, dataset.guidance, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", help="answer this one message and exit")
    parser.add_argument("--backend", choices=["local", "gemini"], default="local")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="below this probability the bot asks for more detail (default 0.5)")
    # Measured: real injury messages had >= 73% known words, off-topic chat mostly <= 50%.
    parser.add_argument("--min-known-words", type=float, default=0.6,
                        help="below this share of words seen in training, the message is treated as off-topic")
    return parser.parse_args()


def load_classifier(backend: str, checkpoint: Path, labels: list[str]) -> Classifier:
    """The chosen backend, or a clean exit with the reason if it cannot be loaded."""
    print("Loading model...")
    try:
        return build_classifier(backend, checkpoint, labels)
    except (FileNotFoundError, NotImplementedError) as error:
        raise SystemExit(str(error))


def build_classifier(backend: str, checkpoint: Path, labels: list[str]) -> Classifier:
    if backend == "gemini":
        return GeminiClassifier(labels)
    return LocalClassifier(checkpoint)


# ---- the conversation ----

def chat_loop(classifier: Classifier, guidance: dict[str, Guidance], args: argparse.Namespace) -> None:
    """Answer messages until the user types 'quit' or presses Ctrl+C / Ctrl+D."""
    print(f"\nFirst-aid assistant\n{DISCLAIMER}\n{HELP}")
    while True:
        text = read_message()
        if text is None:
            break
        if not text:
            continue
        if text.lower() in QUIT_WORDS:
            break
        if text.lower() in HELP_WORDS:
            print(HELP)
            continue
        print_answer(text, classifier, guidance, args)
    print("Take care. In an emergency call 112.")


def read_message() -> str | None:
    """The next message the user types, or None when the input ends (Ctrl+C or Ctrl+D)."""
    try:
        return input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def print_answer(text: str, classifier: Classifier, guidance: dict[str, Guidance],
                 args: argparse.Namespace) -> None:
    prediction = classifier.predict(text)
    reply = format_answer(prediction, guidance, args.min_confidence, args.min_known_words)
    print("\n" + textwrap.indent(reply, "  "))


# ---- the reply ----

def format_answer(prediction: Prediction, guidance: dict[str, Guidance],
                  min_confidence: float, min_known_words: float) -> str:
    """The bot's reply: off-topic, unsure (ask for more detail), or the first-aid steps."""
    if is_off_topic(prediction, min_known_words):
        return off_topic_reply(guidance)
    if prediction.confidence < min_confidence:
        return unsure_reply(prediction, guidance)
    return first_aid_reply(prediction, guidance)


def is_off_topic(prediction: Prediction, min_known_words: float) -> bool:
    """Too few of the message's words appeared in training. A backend that cannot tell (None) is never off-topic."""
    if prediction.known_words is None:
        return False
    return prediction.known_words < min_known_words


def off_topic_reply(guidance: dict[str, Guidance]) -> str:
    labels = ", ".join(guidance)
    return (f"That does not sound like a situation I know about. I can help with: {labels}.\n"
            "Please describe what happened. If someone is seriously hurt, call 112.")


def unsure_reply(prediction: Prediction, guidance: dict[str, Guidance]) -> str:
    """Show the best guesses and ask for more detail. Warn first if any likely guess is an emergency."""
    lines = []
    emergencies = likely_emergencies(prediction, guidance)
    if emergencies:
        lines.append(f"!!! If this could be {' or '.join(emergencies)}: CALL 112 NOW !!!")
    lines.append(f"I'm not sure what happened. My best guesses: {best_guesses(prediction)}.")
    lines.append("Can you describe it in more detail (what happened, where it hurts, how it looks)?")
    return "\n".join(lines)


def likely_emergencies(prediction: Prediction, guidance: dict[str, Guidance]) -> list[str]:
    """Emergency labels among the top guesses that are at least CAUTION_PROBABILITY likely."""
    emergencies = []
    for label, probability in prediction.ranked[:TOP_GUESSES]:
        if probability >= CAUTION_PROBABILITY and guidance[label].is_emergency:
            emergencies.append(label)
    return emergencies


def best_guesses(prediction: Prediction) -> str:
    """'burn (45%), small cut (30%), bruise (10%)'"""
    guesses = []
    for label, probability in prediction.ranked[:TOP_GUESSES]:
        guesses.append(f"{label} ({probability:.0%})")
    return ", ".join(guesses)


def first_aid_reply(prediction: Prediction, guidance: dict[str, Guidance]) -> str:
    """'CALL 112' first if it is an emergency, then the label, warnings, kit items, steps and source."""
    top = guidance[prediction.label]
    lines = []
    if top.is_emergency:
        lines.append("!!! CALL 112 NOW !!!")
    lines.append(f"This looks like: {top.label.upper()}  ({prediction.confidence:.0%} sure, urgency: {top.urgency})")
    lines += more_urgent_warnings(prediction, guidance)
    lines.append(kit_line(top))
    lines += steps_lines(top)
    if top.source:
        lines.append(f"Source: {top.source}")
    return "\n".join(lines)


def more_urgent_warnings(prediction: Prediction, guidance: dict[str, Guidance]) -> list[str]:
    """A warning for each runner-up guess that is fairly likely AND more urgent than the top guess."""
    top = guidance[prediction.label]
    warnings = []
    for label, probability in prediction.ranked[1:TOP_GUESSES]:
        alternative = guidance[label]
        if probability >= CAUTION_PROBABILITY and alternative.urgency_rank > top.urgency_rank:
            warnings.append(warning_line(alternative, probability))
    return warnings


def warning_line(alternative: Guidance, probability: float) -> str:
    what_to_do = "follow that advice"
    if alternative.is_emergency:
        what_to_do = "follow that advice and call 112"
    return (f"Careful: it could also be {alternative.label} ({probability:.0%}), which is '{alternative.urgency}'. "
            f"If that fits better, {what_to_do}.")


def kit_line(entry: Guidance) -> str:
    """'Kit items: #1 plasters, #2 gauze' or 'Kit items: none needed'."""
    if not entry.kit_items:
        return "Kit items: none needed"
    items = []
    for item in entry.kit_items:
        items.append(f"#{item.item_id} {item.name}")
    return "Kit items: " + ", ".join(items)


def steps_lines(entry: Guidance) -> list[str]:
    """'Steps:' and then each step, wrapped to 90 characters."""
    lines = ["Steps:"]
    for step in entry.steps.splitlines():
        lines.append(textwrap.fill(step.strip(), width=90, initial_indent="  ", subsequent_indent="     "))
    return lines


if __name__ == "__main__":
    main()
