# First-Aid Assistant (prototype)

> **Educational prototype only.** This is not medical advice and does not replace a doctor.

A terminal chatbot. You describe an injury ("I touched the hot stove"). It works out which of 11
first-aid situations it is, then answers with:
- the urgency;
- the kit items to use (by ID);
- the British Red Cross steps from `scenarios_redcross.xlsx`.

**The whole AI is trained by you, from scratch, on your own sentences.**
- No pretrained model is used. 
- The only libraries are:
  - PyTorch (to build and train the network);
  - scikit-learn (cross-validation splits and scores);
  - NumPy;
  - openpyxl (to read the Excel file).
- Every number the model uses comes from training on the sentences in Sheet1. That includes its
  vocabulary, its embeddings, its weights and its biases.

The chatbot talks to the model through an interface. A Gemini Flash-Lite backend can be added
later without changing the chatbot.

---

## How it works

```
your message: "I touched a hot pot"
  │
  ├─ 1. VOCABULARY     split into features learned from YOUR training sentences:
  │                      words      i, touched, a, hot, pot
  │                      word pairs i_touched, touched_a, a_hot, hot_pot
  │                      pieces     <ho, hot, ot>, <po, pot, ...   (3-5 letters of each word)
  │                    → a list of feature IDs (features never seen in training are ignored)
  │
  ├─ 2. EMBEDDINGS     the network's embedding layer holds a learned vector for every feature;
  │                    the sentence embedding is the average of its features' vectors
  │
  ├─ 3. NEURAL NETWORK sentence embedding → hidden layer (ReLU, dropout) → 11 scores
  │                    → softmax → a probability per label
  │
  ├─ 4. SAFETY CHECKS  • off-topic?  under 60% of the words were seen in training → ask what happened
  │                    • unsure?     top probability under 50% → show best guesses, ask for detail
  │                    • a more urgent runner-up? → "Careful: it could also be …"
  │
  └─ 5. LOOKUP         Sheet2: steps, urgency, kit_item_ids   Sheet3: kit item names
                        → answer. Emergencies always start with "!!! CALL 112 NOW !!!"
```

Steps 1–3 are all learned during training. The word pieces let the model link words it has never
seen in full: "scalding" and "scalded" share `<sc`, `sca`, `cal`, `ald`, …

| Step from the assignment | Where in the code |
|---|---|
| 1. Convert scenario text into embeddings | `firstaid/vocabulary.py` (text → IDs), `firstaid/model.py` (learned embedding layer) |
| 2. Basic neural network in PyTorch | `firstaid/model.py` |
| 3. Hyperparameters, an algorithm that tunes them, weights and biases being adjusted | `firstaid/training.py` |
| 4. Train it and give feedback | `train.py` + `firstaid/evaluation.py` |
| Terminal chatbot | `chat.py` |
| Gemini later | `firstaid/backends.py` (`GeminiClassifier`) |

## Project layout

```
AiDev-no-mpnet/
├─ scenarios_redcross.xlsx   the data (Sheet1 scenarios, Sheet2 guidance, Sheet3 kit items, Notes)
├─ train.py                  run this to train: tune → train → feedback → save
├─ chat.py                   run this to chat with the trained model
├─ requirements.txt          Python packages (exact versions that were tested)
├─ firstaid/
│  ├─ data.py                reads the workbook by column name and checks it
│  ├─ vocabulary.py          text → features → IDs; vocabulary built from training sentences
│  ├─ model.py               the network (FirstAidNet): embedding layer + hidden layers; save/load
│  ├─ training.py            HyperParams, SEARCH_SPACE, training loop, cross-validation, random search
│  ├─ evaluation.py          the feedback report
│  └─ backends.py            Classifier interface, LocalClassifier, GeminiClassifier placeholder
└─ artifacts/                created by train.py
   ├─ model.pt               your trained model: vocabulary, embeddings, weights, biases, hyperparameters
   ├─ training_report.txt    the feedback from the last training run
   └─ tuning_trials.json     every hyperparameter combination tried, with its score
```

---

## Setup (once)

You need Python 3.10 or newer. It was tested on 3.14 and Windows 11. In PowerShell, from the
`AiDev-no-mpnet` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then activate the environment, so that `python` means the project's Python:

```powershell
.\.venv\Scripts\Activate.ps1
# if PowerShell refuses ("running scripts is disabled"), run this once, then try again:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(Alternatively, skip activating and type `.\.venv\Scripts\python.exe` instead of `python`.)

---

## Training it yourself, step by step

### 1. Check your data

Open `scenarios_redcross.xlsx`.
- **Sheet1** holds the training examples. Each row has:
  - `scenario_id`: a unique number;
  - `use_case_description`: the sentence;
  - `label`: one of the 11 labels in Sheet2;
  - `split`: `train` or `test`.
- **The split rule** (from Notes): if `scenario_id` is divisible by 5 it is `test`, otherwise
  `train`.
  - The model learns only from `train`.
  - `test` sentences are kept hidden to check it honestly at the end.
- **Sheet2 and Sheet3** are what the chatbot answers with. They are not used for training.

Save the workbook. It may stay open in Excel, but unsaved changes are not seen.

### 2. Run the training

```powershell
python train.py
```

This takes about 4–5 minutes on a laptop CPU. You will see 4 stages.

**Stage 1: Loading data.** It reads the workbook and checks that every label and kit ID matches up.
```
1) Loading data
   110 scenarios, 11 labels: 88 train / 22 test
```

**Stage 2: Tuning hyperparameters.** It tries 40 combinations of settings. Each one is scored by
cross-validation: train on 3/4 of the training sentences, test on the other 1/4, four times, and
average. The best one wins.
```
   trial   1/40   cv accuracy  63.6% +/- 8.5%   emb=64 pieces=on hidden=64x1 drop=0.2 lr=0.01 ...
   ...
   Top 5:
   1. cv accuracy  70.5%  loss 1.161   emb=128 pieces=on hidden=128x1 drop=0.0 lr=0.01 ...
```

**Stage 3: Training the final network.** It builds the vocabulary from your 88 train sentences,
then trains from random starting weights.
- **loss** = how wrong the model is. It should go down.
- **train accuracy** = how many training sentences it gets right.
```
    epoch   10/100   loss 0.0002   train accuracy 100.0%
    ...
   vocabulary: 4,249 features found in your sentences
   network: embeddings (4,249 x 128) -> hidden 128 -> 11 labels
   learned: 561,664 weights + 139 biases
```

**Stage 4: Feedback.** It tests on the 22 hidden test sentences. See
[Reading the feedback](#3-read-the-feedback).

Finally it saves `artifacts/model.pt`, which the chatbot loads.

### 3. Read the feedback

- **Accuracy**: training vs cross-validation vs test.
  - Training at 100% but test much lower means the model memorised the examples instead of
    learning the pattern (**overfitting**). With only 8 sentences per label this is expected.
  - If even training accuracy is low, it hasn't learned enough (**underfitting**).
- **Per label**:
  - **Precision**: when it says "burn", how often is it really a burn?
  - **Recall**: of all real burns, how many did it find?
- **Confusion matrix**: which labels get mixed up with which.
- **Mistakes**: every test sentence it got wrong, with how sure it was.
  - **[UNDER-TRIAGE]** marks the dangerous mistakes: something predicted as *less urgent* than it
    is (e.g. choking predicted as insect sting).
- **Advice**: automatic tips based on the numbers above.

### 4. Improve it, then train again

The model knows only the words in your training sentences, so **more and more varied sentences is
by far the biggest improvement**. Changing the network helps much less.

1. Add rows to Sheet1: the next `scenario_id`, a sentence, a label from Sheet2, and `split` by the
   rule above.
2. Save the workbook and run `python train.py` again.
3. Compare the new feedback with `artifacts/training_report.txt` from before. Copy it somewhere
   first, because training overwrites it.

What to add:
- **Sentences the chatbot got wrong.** These are the best new examples.
- **Different words for the same thing.** None of the burn sentences say "burn" or "burned",
  so "I burned my hand on the oven" is currently classified as head injury. Add sentences like
  that. Small cut also needs "scraped", "grazed" and "scratched".
- **Different people and body parts**: "my dad", "a kid at school", "my ankle", "my wrist".
- **Keep the labels balanced.** Add about the same number for each label.

A new label needs a row in Sheet2 too, with its steps, urgency and kit item IDs.

### Training options

| Command | What it does |
|---|---|
| `python train.py --trials 80` | try more combinations (slower, sometimes better) |
| `python train.py --no-tune` | skip tuning, train once with the default hyperparameters (a few seconds) |
| `python train.py --seed 7` | different random start. The same seed always gives the same result, so try a few seeds to see whether a result is luck |
| `python train.py --folds 4` | number of cross-validation folds (4 fits the 8 train sentences per label) |
| `python train.py --data other.xlsx --out other_artifacts` | different workbook / output folder |

To change *which* values are tried, edit `SEARCH_SPACE` in `firstaid/training.py`.
To change the defaults used by `--no-tune`, edit `HyperParams` in the same file.

---

## How to chat

```powershell
python chat.py                                   # interactive; type 'quit' to leave
python chat.py "my friend is choking on bread"   # one message, then exit
```

Example:

```
you> my friend is choking on bread and cannot talk

  !!! CALL 112 NOW !!!
  This looks like: CHOKING  (100% sure, urgency: emergency)
  Kit items: none needed
  Steps:
    1. Encourage them to cough.
    2. If they cannot speak, breathe or cough: bend them forwards and give up to 5 back
       blows between the shoulder blades with the heel of your hand.
    ...
  Source: https://www.redcross.org.uk/first-aid/learn-first-aid/choking
```

| Option | Default | Meaning |
|---|---|---|
| `--min-confidence` | 0.5 | below this probability the bot says it is unsure and shows its best guesses |
| `--min-known-words` | 0.6 | if fewer of the message's words than this appeared in training, it is treated as off-topic |
| `--backend` | `local` | `gemini` is reserved for later (see below) |

**Why the 0.6 threshold?** It was measured, not guessed:
- Real injury messages had at least 73% known words.
- Off-topic messages had at most 50%. Examples: "hello" (0%), "tell me a joke" (50%),
  "what is the weather today" (40%).
- The exception is "my computer is broken". All its words appear in your sentences, so it gets
  through.
- Confidence alone cannot catch off-topic messages: the network was 100% sure "hello" was a head
  injury.

---

## Hyperparameters and the tuning algorithm

Hyperparameters are the settings *you* choose. The network does not learn them. They live in
`HyperParams` in `firstaid/training.py`:

| Hyperparameter | Default | Tried during tuning | What it controls |
|---|---|---|---|
| `embedding_dim` | 64 | 32, 64, 128 | how many numbers each learned embedding vector has |
| `char_ngrams` | on | on, off | whether 3–5 letter word pieces are used as features |
| `hidden_size` | 64 | 32, 64, 128 | neurons per hidden layer |
| `num_hidden_layers` | 1 | 1, 2 | how deep the network is |
| `dropout` | 0.2 | 0.0, 0.2, 0.4 | share of neurons randomly switched off while training (fights overfitting) |
| `learning_rate` | 0.01 | 0.003, 0.01, 0.03 | size of each weight/bias adjustment |
| `weight_decay` | 0.0001 | 0, 0.0001, 0.001 | pulls weights towards zero (fights overfitting) |
| `epochs` | 100 | 50, 100, 200 | passes over the training data |
| `batch_size` | 8 | 8, 16 | sentences per adjustment |

**The tuning algorithm** (`random_search` + `cross_validate` in `training.py`):

1. Trial 1 is always the defaults, as a baseline. The other trials are random picks from the
   table, never the same combination twice.
2. Each combination is scored with **4-fold stratified cross-validation** on the 88 train sentences:
   - split them into 4 groups (2 sentences per label in each);
   - build the vocabulary and train on 3 groups, then test on the 4th;
   - repeat 4 times and average.

   The 22 test sentences are never touched during tuning.
3. Ranking: highest cross-validation accuracy first. Ties go to the lowest cross-validation
   loss (more confident when right).
4. The winner is trained on all 88 train sentences and checked once on the test set.

## Weights and biases

The model learns three kinds of numbers. All of them start random (or zero) and are adjusted by
training:

| What | Where | Starts as | In the current model |
|---|---|---|---|
| **Embedding vectors**: one learned vector per vocabulary feature | `nn.EmbeddingBag` | small random numbers (normal, std 0.1) | 4,249 features × 128 = 543,872 |
| **Weights** of each layer, the matrix `W` in `output = input × Wᵀ + b` | `nn.Linear` | small random numbers (Kaiming init, scaled for ReLU) | 17,792 |
| **Biases** of each layer, the vector `b` | `nn.Linear` | 0 | 139 |

All three are set up in `FirstAidNet.reset_parameters()` in `model.py`. Every training step in
`fit()` in `training.py` then does this:

1. **Forward pass**: run a batch of sentences through the network → 11 scores per sentence.
2. **Loss**: cross-entropy, i.e. how wrong the scores are compared with the true labels.
3. **`loss.backward()`** (backpropagation): calculate, for every embedding number, weight and
   bias, which direction lowers the loss.
4. **`optimizer.step()`** (AdamW): move all of them a small step in that direction.
   `learning_rate` sets the step size.

After training, they are saved in `artifacts/model.pt`.

---

## Results of the current model (30 Oct 2026, seeds 42, 7, 123)

Tuned with 40 trials on the 88 train / 22 test split.

| | Seed 42 (saved model) | Seed 7 | Seed 123 |
|---|---|---|---|
| Cross-validation accuracy | 70.5% | 72.7% | 72.7% |
| **Test accuracy** | **77.3%** (17 / 22) | 72.7% (16 / 22) | 72.7% (16 / 22) |
| Under-triage on test | 1 | 1 | 1 |

The average test accuracy over the three seeds is 74.2%.

The saved model's 5 test mistakes:
- **Under-triage**: "Something got stuck in my throat while I was eating and I am coughing hard"
  (choking) predicted as insect sting;
- small cut ×2 ("scraped my knee", "rose thorn scratched");
- bruise → broken bone;
- sprain → severe allergic reaction.

The common cause is words the model never saw in training: "scraped", "thorn", "stuck", "throat",
"landed". Adding sentences with those words is the fix (see step 4).

## Known limitations

- **It only knows the words in your training sentences.** Unknown words are ignored, apart from
  the word pieces they share with known words.
- **Overconfident.** It is often 90–100% sure even when wrong, because it memorises the 88
  sentences. Do not read the percentage as a real probability.
- **One situation per message.** "I cut my hand and burned it" picks only one.
- **English only.**
- **Tiny test set.** It has 22 sentences, so one mistake moves accuracy by 4.5%. Use `--seed` to
  see how stable a result is.

---

## Switching to Gemini later

`chat.py` only talks to the `Classifier` interface in `firstaid/backends.py`:

```python
class Classifier(ABC):
    def predict(self, text: str) -> Prediction: ...   # Prediction.ranked = [(label, probability), ...]
```

`LocalClassifier` is your trained model. `GeminiClassifier` is a **placeholder**. Right now
`python chat.py --backend gemini` just says it is not implemented. To make it work:

1. `pip install google-genai` and set the `GEMINI_API_KEY` environment variable.
2. In `GeminiClassifier.predict()`, send Gemini Flash-Lite a prompt with the list of labels and the
   user's message. Ask for exactly one label back; structured JSON output is the most reliable.
3. Check that the answer is one of the labels and return `Prediction([(label, 1.0)])`.

Everything else (Sheet2/Sheet3 lookup, "CALL 112" first, kit items) keeps working unchanged.
A sensible combination is to use your local model first and ask Gemini only when the local model
is unsure or the message is off-topic.

---

## Changes made to `scenarios_redcross.xlsx`

- **Sheet2 column B**: was `kit_items` (item names), now `kit_item_ids`. It holds the Sheet3 IDs
  each label needs, e.g. small cut = `1, 2, 10, 11`. The cell is empty when no kit item is needed
  (choking, severe allergic reaction).
- **Notes**:
  - the kit inventory now says 11 items and includes `plastic card: insect sting`;
  - the plastic card was removed from the "not in the kit" line;
  - a line explains Sheet2 column B.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No trained model at …\artifacts\model.pt` | run `python train.py` first |
| `Sheet1 scenario N has label 'x' that is not in Sheet2` | typo or trailing space in the label; it must match Sheet2 exactly |
| `Sheet2 'x' uses kit item id(s) [..] that are not in Sheet3` | add the item to Sheet3 or fix the ID |
| New sentences have no effect | save the workbook in Excel, then retrain |
| `ModuleNotFoundError: torch` | you are not using the venv: activate it, or run `.\.venv\Scripts\python.exe` |
| The bot says a real injury is off-topic | the message uses words it has never seen: add sentences with those words to Sheet1, or lower `--min-known-words` |
