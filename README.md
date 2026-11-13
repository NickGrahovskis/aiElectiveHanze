# First-Aid Assistant (prototype)

> **Educational prototype only.** This is not medical advice and does not replace a doctor.

A terminal chatbot. You describe an injury, and it answers with the urgency, the kit items to use
and the British Red Cross steps. A PyTorch network is trained on the sentences in
`scenarios_redcross1.xlsx`. For how the model works, see `MODEL_EXPLAINED.md`.

## Structure

```
aiElectiveHanze/
├─ scenarios_redcross1.xlsx  data: Sheet1 scenarios (train/test), Sheet2 guidance, Sheet3 kit items
├─ train.py                  tune → train → feedback → save
├─ chat.py                   chat with the trained model
├─ requirements.txt
├─ firstaid/
│  ├─ data.py                reads and checks the workbook
│  ├─ vocabulary.py          text → words, word pairs, word pieces → IDs
│  ├─ mpnet.py               pre-trained mpnet sentence model (Hugging Face)
│  ├─ model.py               the network (FirstAidNet); save/load
│  ├─ training.py            HyperParams, SEARCH_SPACE, training loop, cross-validation, random search
│  ├─ evaluation.py          feedback report
│  └─ backends.py            loads model.pt and predicts for the chatbot
└─ artifacts/                written by train.py
   ├─ model.pt               trained model
   ├─ training_report.txt    feedback from the last run
   └─ tuning_trials.json     every hyperparameter combination tried
```

## Setup

Python 3.10+ (tested on 3.14, Windows 11). In PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
# if scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The first run with mpnet downloads about 420 MB (cached afterwards).

## Running (every new terminal)

Activate the project environment before running anything. If the prompt starts with `(base)`
(Anaconda), leave it first, otherwise `python` is Anaconda's and fails with
`ModuleNotFoundError: No module named 'sklearn'`:

```powershell
conda deactivate
.\.venv\Scripts\Activate.ps1
python train.py
```

The prompt should now start with `(.venv)`. Without activating, `.\.venv\Scripts\python.exe train.py`
works too.

## Commands

```powershell
python train.py                                  # tune, train, evaluate, save to artifacts/
python chat.py                                   # interactive chat; type 'quit' to leave
python chat.py "my friend is choking on bread"   # answer one message and exit
```

## Training options (`train.py`)

| Option | Default | What it does |
|---|---|---|
| `--trials N` | 40 | hyperparameter combinations to try |
| `--folds N` | 4 | cross-validation folds per trial |
| `--seed N` | 42 | random seed (same seed = same result) |
| `--no-tune` | off | skip tuning, train once with the defaults in `HyperParams` |
| `--mpnet` / `--no-mpnet` | on | use mpnet sentence vectors; `--no-mpnet` trains fully from scratch |
| `--own-embeddings` / `--no-own-embeddings` | off | with mpnet, also learn embeddings from your sentences and use both (mpnet + own); without it, mpnet only. No effect with `--no-mpnet` |
| `--data FILE` | `scenarios_redcross1.xlsx` | workbook to train on |
| `--out DIR` | `artifacts` | where to save model and report |

To change which values tuning tries, edit `SEARCH_SPACE` in `firstaid/training.py`. To change the
defaults (including `label_smoothing`, which is not tuned), edit `HyperParams` in the same file.

## Reading the training report (`artifacts/training_report.txt`)

`train.py` keeps the sentences marked `test` in Sheet1 aside and never trains on them. The report
shows how the model does on those unseen sentences, which is the honest measure of quality.

### Accuracy

| Line | Meaning |
|---|---|
| training data | score on the sentences it learned from; near 100% is normal and proves little |
| cross-validation | estimate made during tuning, on parts of the training split held back in turn |
| test data | share of unseen test sentences labelled correctly; **the number that matters** |

A big gap between training (high) and test (low) means the model memorised its sentences instead of
learning the pattern (overfitting).

### Per label (test set)

| Column | Question it answers | Formula |
|---|---|---|
| precision | When the model says X, how often is it right? | correct X ÷ times it predicted X |
| recall | Of all sentences that really are X, how many did it find? | correct X ÷ sentences that really are X |
| sentences | how many test sentences have this label | |

- Low **recall**: the model misses this label and calls it something else.
- Low **precision**: the model uses this label too often, for sentences that belong elsewhere.
- For emergencies (severe bleeding, choking, allergic reaction, broken bone, head injury) **recall
  matters most**: a missed emergency is worse than a minor case flagged as urgent.
- With few test sentences per label, one mistake moves a score a lot (with 2 sentences, recall can
  only be 0%, 50% or 100%).

### Confusion matrix

```
                     0  1  2  3 ...
   0 burn            6  .  .  .        rows    = the true label
   1 small cut       1  4  .  .        columns = what the model predicted (same numbers as the rows)
   3 sprain          .  .  .  5  ...   . means 0
```

- Each number is a **count of test sentences**: row X, column Y = "really X, predicted as Y".
- The **diagonal** (top-left to bottom-right) holds the correct answers.
- Anything **off the diagonal** is a mistake and shows which labels get mixed up. Above, one small
  cut was predicted as burn (row 1, column 0).
- Reading **across a row** gives recall; reading **down a column** gives precision.
- Accuracy = sum of the diagonal ÷ sum of all cells.

### Mistakes on the test set

Every wrong test sentence with its true and predicted label and how sure the model was. Lines marked
`[UNDER-TRIAGE]` are the dangerous ones: the prediction is **less urgent** than reality (for example a
broken bone answered as a sprain, self-care instead of see a doctor). Fix these first, usually by
adding more varied example sentences for the two labels it confuses.

## Chat options (`chat.py`)

| Option | Default | What it does |
|---|---|---|
| `--min-confidence` | 0.5 | below this probability the bot says it is unsure, shows its likely guesses and asks what happened, where on the body and what the person feels (`unknown` and `clarification` always get this reply, with no urgency line) |
| `--min-known-words` | 0.6 | if fewer of the message's words than this were seen in training, it is treated as off-topic |
| `--fix-spelling` / `--no-fix-spelling` | on | fix typos with training words before the model reads them (`my irm is brokem` -> `my arm is broken`) and show `(read as: ...)`; see `firstaid/spelling.py` |
| `--checkpoint FILE` | `artifacts/model.pt` | model to load |
| `--data FILE` | `scenarios_redcross1.xlsx` | workbook with the guidance (Sheet2/3) |
