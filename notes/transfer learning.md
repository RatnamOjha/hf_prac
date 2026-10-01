## Transfer learning

Source: [What is Transfer Learning?](https://www.youtube.com/watch?v=BqqfQnyjmgg)

**Idea:** reuse what a model learned on a big task (A) to solve a new task (B).

| Approach | Starting weights | Cost | Result |
|---|---|---|---|
| Train from scratch | random | lots of data, time, compute | worse |
| Transfer learning (fine-tune) | copied from pretrained model A | little data, fast | better |

### Proof: BERT on "are these 2 sentences similar?"

| Method | Accuracy |
|---|---|
| From scratch | ~70% (stays there even with longer training) |
| Fine-tuned pretrained | 86%+ |

Why: pretraining on huge text gives the model a statistical understanding of the language.

### Computer vision vs NLP pretraining

| | Computer vision | NLP |
|---|---|---|
| Used for | ~10 years | more recent |
| Typical data | ImageNet: 1.2M images, 1000 labels | huge amounts of raw text |
| Learning type | **supervised** (humans label data) | **self-supervised** (labels come from the text itself) |

### Self-supervised pretraining objectives (NLP)

| Objective | What the model does | Example model | Pretraining data |
|---|---|---|---|
| Next-word prediction | guess the next word | GPT-2 | text from 45M Reddit links |
| Masked language modeling | fill in hidden words (like fill-in-the-blank) | BERT | English Wikipedia + 11,000 unpublished books |

### How fine-tuning works

| Part | What happens |
|---|---|
| Body (pretrained layers) | keep it |
| Head (last layers, built for the pretraining task) | throw it away |
| New head | add one, randomly initialized, sized for your task |

Example: BERT's masked-word head → replaced with a classifier with 2 outputs (2 labels).

**Rule:** pick a pretrained model close to your task.
Classifying German sentences → use a German pretrained model.

### Downside: bias transfers too

| Model | Bias |
|---|---|
| ImageNet models | mostly US and Western Europe images → work better on images from there |
| GPT-3 | "He was very…" → mostly neutral adjectives; "She was very…" → mostly physical ones |
| GPT-2 | OpenAI's model card admits bias and advises against using it in systems that interact with humans |

### Summary

pretrained model (big data) → remove head → add new head → fine-tune on small data → better results, less cost, **same biases**
