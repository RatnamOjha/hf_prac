
| Cell | Task | what it does |
|---|---|---|
| 2 | `sentiment-analysis` | if the Sentence is positive or negative |
| 3 | `zero-shot-classification` | give labels -> classifies words on label |
| 5 | `text-generation` | complete the sentence |
| 6 | `fill-mask` | `<mask>` field gets filled |
| 7 | `ner` | name, entity recognization |

## What happens inside `pipeline()`

The pipeline has three stages:

```
Raw text ──► Tokenizer ──► Input IDs ──► Model ──► Logits ──► Postprocessing ──► Predictions
```

| Stage | Input → Output | Example |
|---|---|---|
| Tokenizer | raw text → input IDs | `"This course is amazing!"` → `[101, 2023, 2607, 2003, 6429, 999, 102]` |
| Model | input IDs → logits | → `[-4.3630, 4.6859]` |
| Postprocessing | logits → predictions (softmax) | → `POSITIVE: 99.89%`, `NEGATIVE: 0.11%` |

### step 1: tokenizer
`tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")`

- since a model can only understand numbers, the tokenizer splits each sentence into words/pieces. 
101 and 102 are special tokens added automatically: [CLS] marks the start [SEP] marks the end. 

- padding = True: it means that if we give 2 sentences as input and one of them is shorter than the other, we apply padding to match the length of longest input(zero-padding).

- truncation = True: means it cuts off any sentence longer than model's limit.

- return_tensors="pt": gives PyTorch tensors instead of Python lists.

- attention_mask: 1 means a real token and 0 means padding, so the model ignores the padding.

### step 2: AutoModel 
`model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)        # ** unpacks the dict → input_ids=..., attention_mask=...
outputs.last_hidden_state.shape  # torch.Size([2, 16, 768])`

it loads the transformer without its task-specific head. 


---

## Doing the pipeline by hand, step by step

Think of it like this: the model is a really smart friend who only speaks numbers.
So we (1) translate our words into numbers, (2) let the friend think, (3) translate
their answer back into something we understand.

Example sentences used below:
- `"I've been waiting for a HuggingFace course my whole life."` (clearly happy)
- `"I hate this so much!"` (clearly not happy)

### Step 1: Tokenizer: turn words into numbers

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
```

The tokenizer chops each sentence into small pieces (tokens) and swaps every piece
for its ID number from the model's dictionary. You get back two things:

**`input_ids`**: the sentences as numbers:
```
[101, 1045, 1005, 2310, ..., 1012, 102]   ← sentence 1 (16 tokens)
[101, 1045, 5223, 2023, 2061, 2172, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0]   ← sentence 2
```
- `101` = "start of sentence" and `102` = "end of sentence". The tokenizer adds these by itself.
- Sentence 2 is shorter, so it gets topped up with `0`s. That's **padding**. The model
  needs every row to be the same length, like rows in a spreadsheet.

**`attention_mask`**: a "pay attention here" sheet:
```
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
```
`1` = real word, read it. `0` = just padding, ignore it.

What the options mean:
| Option | In plain words |
|---|---|
| `padding=True` | make all sentences the same length by adding 0s |
| `truncation=True` | if a sentence is too long for the model, chop off the end |
| `return_tensors="pt"` | give me PyTorch tensors, not plain Python lists |

> ⚠️ Always load the tokenizer and the model from the **same checkpoint**.
> Every model has its own dictionary. Number `2023` means one word to one model
> and something totally different to another.

### Step 2: AutoModel: the "brain" without a mouth

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)   # torch.Size([2, 16, 768])
```

`**inputs` just means "pass everything in the dict": `input_ids=..., attention_mask=...`.

The output shape `[2, 16, 768]` reads as:
**2 sentences × 16 tokens × 768 numbers describing each token.**

So the model *understood* every word, and each word is now a list of 768 numbers
capturing its meaning. But it never told us "positive" or "negative". Plain `AutoModel`
has no **head**, the final layer that turns understanding into an answer.
It's a brain with no mouth.

### Step 3: AutoModelForSequenceClassification: brain + mouth = logits

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits)
# tensor([[-1.5607,  1.6123],
#         [ 4.1692, -3.3464]])
```

Now the model has a **classification head** attached. For every sentence it gives
**one score per label**: here 2 labels, so 2 scores per sentence.

These scores are called **logits**. They are raw "gut feeling" scores:
- bigger number = the model leans that way more
- they can be negative, and they don't add up to anything nice

So they're not percentages yet.

Pick the right "head" for your job:
| Task | Class to use |
|---|---|
| classify a whole sentence (sentiment, spam, topic) | `AutoModelForSequenceClassification` |
| label each word (names, places → NER) | `AutoModelForTokenClassification` |
| find an answer inside a text | `AutoModelForQuestionAnswering` |

### Step 4: Softmax: turn gut feelings into percentages

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# tensor([[4.0195e-02, 9.5980e-01],
#         [9.9946e-01, 5.4418e-04]])
```

**Softmax** squashes the logits into numbers between 0 and 1 that **add up to 1**
in each row. Now they're real probabilities.
(`dim=-1` = "do it across the labels", i.e. the last dimension.)

Reading the result (`e-02` just means "move the decimal 2 places left"):
| Sentence | label 0 | label 1 |
|---|---|---|
| "I've been waiting…" | 0.04 (4%) | 0.96 (96%) |
| "I hate this so much!" | 0.9995 (99.95%) | 0.0005 (0.05%) |

But which label is 0 and which is 1? Ask the model:
```python
model.config.id2label   # {0: 'NEGATIVE', 1: 'POSITIVE'}
```
So sentence 1 → **POSITIVE (96%)**, sentence 2 → **NEGATIVE (99.95%)**. 🎉

---

## Cheat sheet: the recipe for classifying *anything*

Whenever you classify text, it's always these same moves:

1. **Pick a checkpoint** that was trained for your task.
2. **Load the tokenizer** from that checkpoint.
3. **Load the model with the right head** from the *same* checkpoint.
4. **Tokenize**: `padding=True, truncation=True, return_tensors="pt"`.
5. **Run the model** → get logits.
6. **Softmax** → get probabilities.
7. **Pick the biggest one** (`argmax`) and **look up its name** (`id2label`).

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

texts = ["I love this!", "This is terrible."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():                 # we're only predicting, not training, so skip the extra memory
    logits = model(**inputs).logits

probs = torch.softmax(logits, dim=-1)
for text, p in zip(texts, probs):
    best = p.argmax().item()
    print(text, "→", model.config.id2label[best], f"{p[best]:.2%}")
```

One-line memory trick:
**words → numbers → brain → scores → percentages → label**

That's literally all `pipeline()` does for you behind the scenes.