"""Terminal first-aid chatbot.

    python chat.py                                  # interactive chat
    python chat.py "I burned my hand on the oven"   # answer one message and exit
    python chat.py --no-fix-spelling                # read messages exactly as typed (no typo fixing)
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from firstaid.backends import DEFAULT_CHECKPOINT, Classifier, LocalClassifier, Prediction
from firstaid.data import DEFAULT_WORKBOOK, Guidance, load_dataset

DISCLAIMER = ("Educational prototype only. This is not medical advice and does not replace a doctor.\n"
              "In an emergency call 112.")
HELP = "Describe what happened in your own words, e.g. 'I cut my finger while cooking'. Type 'quit' to leave."
QUIT_WORDS = {"quit", "exit", "q"}
HELP_WORDS = {"help", "?"}
TOP_GUESSES = 3             # how many of the most likely labels the bot looks at
CAUTION_PROBABILITY = 0.15  # mention a more urgent runner-up label if it is at least this likely

# Sheet2 labels that mean "I cannot name an injury". They are never shown as a diagnosis with an urgency:
# the bot asks the guided question instead, however sure the model is.
NOT_A_DIAGNOSIS = {"unknown", "clarification"}
# The three things from the Use Case: what happened, where on the body, and what the person feels.
GUIDED_QUESTION = ("To help you, please tell me three things:\n"
                   "  1. What happened?  (e.g. touched something hot, fell, got stung, cut yourself)\n"
                   "  2. Where on the body?  (e.g. hand, ankle, head, face)\n"
                   "  3. What does the person feel?  (e.g. pain, swelling, bleeding, itching, "
                   "trouble breathing)")
CALL_112_SIGNS = ("If someone is not breathing, unconscious, bleeding heavily, has a swelling face or throat, "
                  "chest pain or a seizure: call 112 now.")


def main() -> None:
    options = parse_args()
    dataset = load_dataset(options.data)
    classifier = load_classifier(options.checkpoint)

    if options.message:
        print_answer(options.message, classifier, dataset.guidance, options)
        return
    chat_loop(classifier, dataset.guidance, options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("message", nargs="?", help="answer this one message and exit")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="below this probability the bot asks for more detail (default 0.5)")
    # Measured: real injury messages had >= 73% known words, off-topic chat mostly <= 50%.
    parser.add_argument("--min-known-words", type=float, default=0.6,
                        help="below this share of words seen in training, the message is treated as off-topic")
    parser.add_argument("--fix-spelling", default=True, action=argparse.BooleanOptionalAction,
                        help="fix typos with words from the training sentences before the model reads them "
                             "(on by default), e.g. 'my irm is brokem' -> 'my arm is broken'. See firstaid/spelling.py")
    return parser.parse_args()


def load_classifier(checkpoint: Path) -> Classifier:
    """The trained model, or a clean exit with the reason if it cannot be loaded."""
    print("Loading model...")
    try:
        return LocalClassifier(checkpoint)
    except FileNotFoundError as error:
        raise SystemExit(str(error))


# ---- the conversation ----

def chat_loop(classifier: Classifier, guidance: dict[str, Guidance], options: argparse.Namespace) -> None:
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
        print_answer(text, classifier, guidance, options)
    print("Take care. In an emergency call 112.")


def read_message() -> str | None:
    """The next message the user types, or None when the input ends (Ctrl+C or Ctrl+D)."""
    try:
        return input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def print_answer(text: str, classifier: Classifier, guidance: dict[str, Guidance],
                 options: argparse.Namespace) -> None:
    if options.fix_spelling:
        text = fix_spelling(text, classifier)
    prediction = classifier.predict(text)
    reply = format_answer(prediction, guidance, options.min_confidence, options.min_known_words)
    print("\n" + textwrap.indent(reply, "  "))


def fix_spelling(text: str, classifier: Classifier) -> str:
    """The message with its typos fixed. If anything changed, show what the bot read, so a wrong fix is visible."""
    fixed = classifier.fix_spelling(text)
    if fixed != text:
        print(f"\n  (read as: {fixed})")
    return fixed


# ---- the reply ----

def format_answer(prediction: Prediction, guidance: dict[str, Guidance],
                  min_confidence: float, min_known_words: float) -> str:
    """The bot's reply: off-topic, unsure (ask the guided question), or the first-aid steps.
    'unknown' and 'clarification' always get the unsure reply: they are not a diagnosis, so they get no urgency."""
    if is_off_topic(prediction, min_known_words):
        return off_topic_reply(guidance)
    if prediction.label in NOT_A_DIAGNOSIS or prediction.confidence < min_confidence:
        return unsure_reply(prediction, guidance)
    return first_aid_reply(prediction, guidance)


def is_off_topic(prediction: Prediction, min_known_words: float) -> bool:
    """Too few of the message's words appeared in training. If it is unknown (None), the message is never treated as off-topic."""
    if prediction.known_words is None:
        return False
    return prediction.known_words < min_known_words


def off_topic_reply(guidance: dict[str, Guidance]) -> str:
    return "\n".join([out_of_scope_line(guidance), GUIDED_QUESTION, CALL_112_SIGNS])


def out_of_scope_line(guidance: dict[str, Guidance]) -> str:
    labels = ", ".join(label for label in guidance if label not in NOT_A_DIAGNOSIS)
    return (f"That does not sound like an injury I know about. I can help with: {labels}.\n"
            "For other health problems contact a doctor or pharmacist.")


def unsure_reply(prediction: Prediction, guidance: dict[str, Guidance]) -> str:
    """No diagnosis: say why, then ask the guided question. Warn first if any likely guess is an emergency."""
    lines = []
    emergencies = likely_emergencies(prediction, guidance)
    if emergencies:
        lines.append(f"!!! If this could be {' or '.join(emergencies)}: CALL 112 NOW !!!")
    lines.append(unsure_opening(prediction, guidance))
    lines.append(GUIDED_QUESTION)
    if not emergencies:
        lines.append(CALL_112_SIGNS)
    return "\n".join(lines)


def unsure_opening(prediction: Prediction, guidance: dict[str, Guidance]) -> str:
    """Why the bot gives no first-aid steps: the message is out of scope, or it is too vague to tell."""
    if prediction.label == "unknown":
        return out_of_scope_line(guidance)
    guesses = best_guesses(prediction)
    if guesses:
        return f"I'm not sure what happened. It might be: {guesses}."
    return "I need a bit more information to know what happened."


def likely_emergencies(prediction: Prediction, guidance: dict[str, Guidance]) -> list[str]:
    """Emergency labels among the top guesses that are at least CAUTION_PROBABILITY likely."""
    emergencies = []
    for label, probability in prediction.ranked[:TOP_GUESSES]:
        if probability >= CAUTION_PROBABILITY and guidance[label].is_emergency:
            emergencies.append(label)
    return emergencies


def best_guesses(prediction: Prediction) -> str:
    """'burn (45%), small cut (30%)': injuries among the top guesses that are at least CAUTION_PROBABILITY likely.
    Empty if there are none, so 'clarification (83%)' or 'bruise (2%)' is never offered as a guess."""
    guesses = []
    for label, probability in prediction.ranked[:TOP_GUESSES]:
        if label not in NOT_A_DIAGNOSIS and probability >= CAUTION_PROBABILITY:
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
