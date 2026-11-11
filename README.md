# First-Aid Assistant (prototype)

> **Educational prototype only.** This is not medical advice and does not replace a doctor.

A terminal chatbot. You describe an injury ("I touched the hot stove"). It works out which of 11
first-aid situations it is, then answers with:
- the urgency;
- the kit items to use (by ID);
- the British Red Cross steps from `scenarios_redcross1.xlsx`.

**How the AI is built.**
- By default, each sentence is turned into numbers by **mpnet**, a pre-trained sentence model from
  Hugging Face. See [The mpnet sentence model](#the-mpnet-sentence-model).
- On top of that sits a small neural network that **you train from scratch** on the sentences in
  Sheet1. Its weights, biases and hyperparameters all come from your data.
- `python train.py --no-mpnet` makes the whole AI from scratch: it also learns its own vocabulary and
  embeddings from your sentences, and no pre-trained model is used.
- Libraries:
  - PyTorch (to build and train the network);
  - scikit-learn (cross-validation splits and scores);
  - NumPy;
  - openpyxl (to read the Excel file);
  - sentence-transformers (to run mpnet).

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
  ├─ 2. EMBEDDINGS     default: mpnet reads the whole sentence → 768 fixed numbers
  │                    --no-mpnet: the network's embedding layer holds a learned vector for every
  │                    feature from step 1; the sentence embedding is the average of those vectors
  │
  ├─ 3. NEURAL NETWORK sentence embedding → hidden layer (ReLU, dropout) → one score per label
  │                    → softmax → a probability per label
  │
  ├─ 4. SAFETY CHECKS  • off-topic?  under 60% of the words were seen in training → ask what happened
  │                    • unsure?     top probability under 50% → show best guesses, ask for detail
  │                    • a more urgent runner-up? → "Careful: it could also be …"
  │
  └─ 5. LOOKUP         Sheet2: steps, urgency, kit_item_ids   Sheet3: kit item names
                        → answer. Emergencies always start with "!!! CALL 112 NOW !!!"
```

- **Step 1** is always built from your sentences. With mpnet it is only used for the off-topic check
  in step 4. With `--no-mpnet` it is also the network's input.
- **Step 3** is always learned during training.
- **Word pieces** (used with `--no-mpnet`) let the model link words it has never seen in full:
  "scalding" and "scalded" share `<sc`, `sca`, `cal`, `ald`, …

| Step from the assignment | Where in the code |
|---|---|
| 1. Convert scenario text into embeddings | `firstaid/mpnet.py` (pre-trained mpnet, default); `firstaid/vocabulary.py` (text → IDs) + `firstaid/model.py` (learned embedding layer) with `--no-mpnet` |
| 2. Basic neural network in PyTorch | `firstaid/model.py` |
| 3. Hyperparameters, an algorithm that tunes them, weights and biases being adjusted | `firstaid/training.py` |
| 4. Train it and give feedback | `train.py` + `firstaid/evaluation.py` |
| Terminal chatbot | `chat.py` + `firstaid/backends.py` |

## Project layout

```
aiElectiveHanze/
├─ scenarios_redcross1.xlsx  the data (Sheet1 scenarios, Sheet2 guidance, Sheet3 kit items)
├─ train.py                  run this to train: tune → train → feedback → save
├─ chat.py                   run this to chat with the trained model
├─ requirements.txt          Python packages (exact versions that were tested)
├─ firstaid/
│  ├─ data.py                reads the workbook by column name and checks it
│  ├─ vocabulary.py          text → features → IDs; vocabulary built from training sentences
│  ├─ mpnet.py               the pre-trained mpnet sentence model from Hugging Face
│  ├─ model.py               the network (FirstAidNet): embedding layer or mpnet input + hidden layers; save/load
│  ├─ training.py            HyperParams, SEARCH_SPACE, training loop, cross-validation, random search
│  ├─ evaluation.py          the feedback report
│  └─ backends.py            LocalClassifier: loads model.pt and predicts labels for the chatbot
└─ artifacts/                created by train.py
   ├─ model.pt               your trained model: vocabulary, weights, biases, hyperparameters
   ├─ training_report.txt    the feedback from the last training run
   └─ tuning_trials.json     every hyperparameter combination tried, with its score
```

---

## Setup (once)

You need Python 3.10 or newer. It was tested on 3.14 and Windows 11. In PowerShell, from the
project folder:

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

The first training run downloads mpnet (about 420 MB). See
[Download and disk space](#download-and-disk-space).

---

## Training it yourself, step by step

### 1. Check your data

Open `scenarios_redcross1.xlsx`.
- **Sheet1** holds the example sentences. Each row has:
  - `scenario_id`: a unique number (training stops with an error if two rows share one);
  - `use_case_description`: the sentence;
  - `label`: one of the labels in Sheet2;
  - `split`: `train` or `test`.
- **The split column decides** what each sentence is used for:
  - the model learns only from `train`;
  - `test` sentences are kept hidden to check it honestly at the end.
- **Right now** there are 332 sentences: 266 train and 66 test. That is 24 train and 6 test per
  label (head injury has 26 train).
- **Sheet2 and Sheet3** are what the chatbot answers with. Sheet2 also defines the list of labels.
  They are not used as training sentences.

Save the workbook. It may stay open in Excel, but unsaved changes are not seen.

### 2. Run the training

```powershell
python train.py
```

You will see 4 stages.

**Stage 1: Loading data.** It reads the workbook and checks that every label and kit ID matches up.
Then it loads mpnet (downloading it the first time).
```
1) Loading data
   332 scenarios, 12 labels: 266 train / 66 test
   loading mpnet from Hugging Face sentence-transformers/all-mpnet-base-v2 (the first time this downloads ~420 MB)
   mpnet sentence model: 768 numbers per sentence
```

**Stage 2: Tuning hyperparameters.** This is the slow part. It tries 40 combinations of settings.
Each one is scored by 8-fold cross-validation:
- train on 7/8 of the training sentences and test on the other 1/8;
- do that 8 times and average.

That is 320 small trainings. The best combination wins.
```
   trial   1/40   cv accuracy  ...   emb=mpnet hidden=64x1 drop=0.2 lr=0.01 ...
   ...
   Top 5:
   1. cv accuracy  94.4%  loss 0.228   emb=mpnet hidden=32x1 drop=0.4 lr=0.003 wd=0 ep=50 bs=8
```

**Stage 3: Training the final network.** It trains the winning combination on all 266 train
sentences, starting from random weights.
- **loss** = how wrong the model is. It should go down.
- **train accuracy** = how many training sentences it gets right.
```
    epoch    5/50   loss …   train accuracy …
    ...
   vocabulary: 8,196 features found in your sentences (only used to spot off-topic messages)
   network: mpnet embeddings (768 fixed numbers) -> hidden 32 -> 12 labels
   learned: 24,960 weights + 44 biases
```

**Stage 4: Feedback.** It tests on the 66 hidden test sentences. See
[Read the feedback](#3-read-the-feedback).

Finally it saves `artifacts/model.pt`, which the chatbot loads.

### 3. Read the feedback

- **Accuracy**: training vs cross-validation vs test.
  - Training at 100% but test much lower means the model memorised the examples instead of
    learning the pattern (**overfitting**).
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

**More and more varied sentences is by far the biggest improvement.** Changing the network helps
much less.

1. Add rows to Sheet1: the next `scenario_id`, a sentence, a label from Sheet2, and `split`.
   Put about 1 in 5 new sentences in `test`.
2. Save the workbook and run `python train.py` again.
3. Compare the new feedback with `artifacts/training_report.txt` from before. Copy it somewhere
   first, because training overwrites it.

What to add:
- **Sentences the chatbot got wrong.** These are the best new examples.
- **Pairs that get mixed up.** Right now that is choking vs severe allergic reaction, and sprain vs
  broken bone. Add sentences that show the difference.
- **Different words for the same thing**: "scraped", "grazed", "scratched" for a small cut.
- **Different people and body parts**: "my dad", "a kid at school", "my ankle", "my wrist".
- **Keep the labels balanced.** Add about the same number for each label.

A new label needs a row in Sheet2 too, with its steps, urgency and kit item IDs.

### Training options

| Command | What it does |
|---|---|
| `python train.py --trials 80` | try more combinations (slower, sometimes better) |
| `python train.py --no-tune` | skip tuning, train once with the default hyperparameters (about 15 seconds) |
| `python train.py --no-mpnet` | train completely from scratch: learned vocabulary and embeddings instead of mpnet |
| `python train.py --seed 7` | different random start. The same seed always gives the same result, so try a few seeds to see whether a result is luck |
| `python train.py --folds 4` | number of cross-validation folds (default 8: about 3 train sentences per label in each fold) |
| `python train.py --data other.xlsx --out other_artifacts` | different workbook / output folder |

To change *which* values are tried, edit `SEARCH_SPACE` in `firstaid/training.py`.
To change the defaults used by `--no-tune`, edit `HyperParams` in the same file.

---

## How to chat

```powershell
python chat.py                                   # interactive; type 'quit' to leave
python chat.py "my friend is choking on bread"   # one message, then exit
```

`chat.py` reads from `model.pt` whether the model was trained with or without mpnet, so it needs no
options for that.

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

**Why the 0.6 threshold?** It was measured on an earlier, 110-sentence version of the workbook:
- Real injury messages had at least 73% known words.
- Off-topic messages had at most 50%. Examples: "hello" (0%), "tell me a joke" (50%),
  "what is the weather today" (40%).
- Confidence alone cannot catch off-topic messages: the network can be very sure about a message
  that has nothing to do with first aid.
- The check is not perfect. All the words in "my computer is broken" appear in the training
  sentences, so it gets through, and the current model answers **choking, CALL 112** (66% sure).

---

## Hyperparameters and the tuning algorithm

Hyperparameters are the settings *you* choose. The network does not learn them. They live in
`HyperParams` in `firstaid/training.py`:

| Hyperparameter | Default | Tried during tuning | What it controls |
|---|---|---|---|
| `mpnet` | on (`--no-mpnet` turns it off) | not searched | use mpnet's fixed sentence vectors instead of learned embeddings |
| `embedding_dim` | 64 | 32, 64, 128 (only with `--no-mpnet`) | how many numbers each learned embedding vector has |
| `char_ngrams` | on | on, off (only with `--no-mpnet`) | whether 3–5 letter word pieces are used as features |
| `hidden_size` | 64 | 32, 64, 128 | neurons per hidden layer |
| `num_hidden_layers` | 1 | 1, 2 | how deep the network is |
| `dropout` | 0.2 | 0.0, 0.2, 0.4 | share of neurons randomly switched off while training (fights overfitting) |
| `learning_rate` | 0.01 | 0.003, 0.01, 0.03 | size of each weight/bias adjustment |
| `weight_decay` | 0.0001 | 0, 0.0001, 0.001 | pulls weights towards zero (fights overfitting) |
| `epochs` | 100 | 50, 100, 200 | passes over the training data |
| `batch_size` | 8 | 8, 16 | sentences per adjustment |

With mpnet, `embedding_dim` and `char_ngrams` are not used, so they are not searched. That leaves
972 possible combinations; tuning tries 40 of them.

**The tuning algorithm** (`random_search` + `cross_validate` in `training.py`):

1. Trial 1 is always the defaults, as a baseline. The other trials are random picks from the
   table, never the same combination twice.
2. Each combination is scored with **8-fold stratified cross-validation** on the 266 train sentences:
   - split them into 8 groups, with every label in every group (about 3 sentences per label);
   - build the vocabulary and train on 7 groups, then test on the 8th;
   - repeat 8 times and average.

   The 66 test sentences are never touched during tuning.
3. Ranking: highest cross-validation accuracy first. Ties go to the lowest cross-validation
   loss (more confident when right).
4. The winner is trained on all 266 train sentences and checked once on the test set.

## Weights and biases

The model learns these numbers. All of them start random (or zero) and are adjusted by training:

| What | Where | Starts as | In the current model |
|---|---|---|---|
| **Embedding vectors**: one learned vector per vocabulary feature (only with `--no-mpnet`) | `nn.EmbeddingBag` | small random numbers (normal, std 0.1) | none: mpnet's vectors are fixed, not learned |
| **Weights** of each layer, the matrix `W` in `output = input × Wᵀ + b` | `nn.Linear` | small random numbers (Kaiming init, scaled for ReLU) | 24,960 (768 × 32 + 32 × 12) |
| **Biases** of each layer, the vector `b` | `nn.Linear` | 0 | 44 (32 + 12) |

All of them are set up in `FirstAidNet.reset_parameters()` in `model.py`. Every training step in
`fit()` in `training.py` then does this:

1. **Forward pass**: run a batch of sentences through the network → one score per label per sentence.
2. **Loss**: cross-entropy, i.e. how wrong the scores are compared with the true labels.
3. **`loss.backward()`** (backpropagation): calculate, for every weight and bias (and embedding
   number), which direction lowers the loss.
4. **`optimizer.step()`** (AdamW): move all of them a small step in that direction.
   `learning_rate` sets the step size.

After training, they are saved in `artifacts/model.pt`.

---

## Results of the current model

**Saved model** (`artifacts/`): mpnet, tuned with 40 trials, seed 42, on 266 train / 66 test.

| | |
|---|---|
| Cross-validation accuracy | 94.4% |
| **Test accuracy** | **93.9%** (62 / 66) |
| Under-triage on test | 0 |
| Chosen hyperparameters | hidden=32x1 drop=0.4 lr=0.003 wd=0 ep=50 bs=8 |

Its 4 test mistakes:
- small cut → insect sting ("A rose thorn scratched my arm while I was gardening…");
- sprain → broken bone ("…landed badly, my foot is tender and swollen");
- choking → severe allergic reaction ("…grabbing his neck after eating a hot dog");
- severe allergic reaction → choking ("…ate a dish with eggs and her voice is hoarse…"), 95% sure.

None of them is under-triage: the guess is always at least as urgent as the truth.

**mpnet vs from scratch**, same settings (default hyperparameters, no tuning, seed 42):

| | `--no-mpnet` (from scratch) | mpnet (default) |
|---|---|---|
| Test accuracy | 65.2% (43 / 66) | 89.4% (59 / 66) |
| Under-triage on test | 8 | 2 |

Other seeds have not been measured on this workbook yet. Run `--seed 7` and `--seed 123` to see how
stable these numbers are. The test set has 66 sentences, so one mistake moves accuracy by 1.5%.

## Known limitations

- **Overconfident.** It can be 95% sure and still wrong. Do not read the percentage as a real
  probability.
- **The `unknown` label is never learned.** Sheet2 has an `unknown` row, but Sheet1 has no
  sentences with that label, so training only ever pushes its score down and it is in practice
  never predicted. Off-topic messages are caught only by
  the known-words check. To use the label, add off-topic sentences labelled `unknown` to Sheet1.
- **Off-topic messages with familiar words get through** (see "my computer is broken" above).
- **One situation per message.** "I cut my hand and burned it" picks only one.
- **English only.**
- **With `--no-mpnet`, it only knows the words in your training sentences.** Unknown words are
  ignored, apart from the word pieces they share with known words.

---

## The mpnet sentence model

**What it is.** [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
(Sentence Transformers) is a transformer: a large neural network that learned English from over a
billion sentence pairs. It reads a whole sentence, word order included, and turns it into 768 numbers
that stand for its meaning. Sentences that mean the same thing get similar numbers even when they
share few words: "I burned my hand" and "my hand has a burn".

**What it changes.**
- In step 2 of [How it works](#how-it-works), the sentence embedding is mpnet's vector of the
  sentence, instead of embeddings learned from your sentences.
- So the model can make sense of words that are not in Sheet1 at all. Word pieces cannot link
  "scalded" to "burn", but mpnet can: "I scalded my arm with boiling tea" is answered as a burn.
- The hidden layers on top are still trained from scratch on Sheet1, and tuning still picks their
  hyperparameters.
- The vocabulary from your sentences is still built, but only for the off-topic check.

**The trade-off.** The embeddings are not learned from your sentences: they come from a downloaded
model that stays fixed. With `--no-mpnet` the model is 100% from scratch. Check that your assignment
allows pre-trained models before you hand in an mpnet model.

### Download and disk space

- The first mpnet run downloads about 420 MB. The version is pinned (`REVISION` in `mpnet.py`), so
  every download is identical.
- Later runs and the chatbot use the cached copy, also offline.
- Each sentence is encoded once and then remembered, so tuning does not slow down because of mpnet.
- The copy is in your Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`). To free the
  space, run `hf cache rm model/sentence-transformers/all-mpnet-base-v2` with the venv active; the
  next mpnet run downloads it again.
- You may see `Warning: You are sending unauthenticated requests to the HF Hub`. It is harmless:
  no Hugging Face account is needed.

---

## About the workbook

- **Sheet1** `scenario_id | use_case_description | label | split`: the example sentences.
- **Sheet2** `label | kit_item_ids | steps | urgency | red_cross_source`: one row per label.
  - `kit_item_ids` holds the Sheet3 IDs each label needs, e.g. small cut = `1, 2, 10, 11`. The
    cell is empty when no kit item is needed (choking, severe allergic reaction).
  - `urgency` must be `self-care`, `see a doctor` or `emergency`.
- **Sheet3** `medic_kit_item_id | medik_kit_item_name`: the 11 kit items. (`medik` is spelled that
  way in the header, and the code looks it up by that exact name.)

Columns are looked up by header name, so you can reorder them in Excel.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No trained model at …\artifacts\model.pt` | run `python train.py` first |
| `Sheet1 scenario N has label 'x' that is not in Sheet2` | typo or trailing space in the label; it must match Sheet2 exactly |
| `Sheet1 has duplicate scenario_id(s) [..]` | give every Sheet1 row its own number |
| `Sheet2 'x' uses kit item id(s) [..] that are not in Sheet3` | add the item to Sheet3 or fix the ID |
| `Sheet2 'x' has unknown urgency 'y'` | use `self-care`, `see a doctor` or `emergency` |
| New sentences have no effect | save the workbook in Excel, then retrain |
| `ModuleNotFoundError: torch` | you are not using the venv: activate it, or run `.\.venv\Scripts\python.exe` |
| The bot says a real injury is off-topic | the message uses words it has never seen: add sentences with those words to Sheet1, or lower `--min-known-words` |
