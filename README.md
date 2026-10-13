# Analogy classification on the YAGO4-19k knowledge graph

> Can a neural network tell whether **a : b :: c : d** is a valid analogy, using only sentence embeddings of the four entities?
> Triples from the YAGO4-19k knowledge graph are turned into valid and invalid analogies, and a CNN analogy classifier (**ANNc**) is trained on top of `bge-large-en-v1.5` embeddings.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-ANNc-ee4c2c) ![HF](https://img.shields.io/badge/Hugging%20Face-bge--large-yellow)

## Idea

Two triples sharing a predicate give an analogy:

```
(Paris, capitalOf, France)  +  (Rome, capitalOf, Italy)   ->   Paris : France :: Rome : Italy   ✔ valid
```

```
triples (s, p, o) ──► split train / val / test BY TRIPLE
                         │
                         ├─► valid analogies   : two true triples, same predicate
                         └─► invalid analogies : (1) corrupted  – head or tail replaced, triple absent from the KG
                                                 (2) cross-relation – two true triples, DIFFERENT predicates
                                    │
                each entity ──► bge-large-en-v1.5 embedding (1024-d, mean pooling)
                                    │
                [a | b | c | d] 1024×4 "image" ──► ANNc (2 conv layers + linear) ──► P(valid)
```

The **cross-relation** negatives are the interesting ones: each pair is a true fact, so the model cannot succeed by checking whether a pair "exists" — it has to compare the two relations, which is what analogy is about.

## Results

*Held-out test sets of 5,000 balanced analogies, built from triples never seen in training. Accuracy (ROC-AUC in brackets). Reproduce with [`colab_run.ipynb`](colab_run.ipynb) on a free T4 GPU.*

| Model | Trained on | Test: corrupted negatives | Test: cross-relation negatives |
|---|---|---|---|
| Parallelogram baseline (no training) | — | 0.60 (0.65) | 0.66 (0.69) |
| ANNc | corrupted negatives | **0.83 (0.94)** | 0.50 (0.50) |
| ANNc | cross-relation negatives | 0.66 (0.73) | **0.77 (0.86)** |

**Key finding.** The model trained like in the original project (corrupted negatives) reaches 83 % accuracy, but it is **at chance level (50 %) on cross-relation negatives**. It has learned to detect *implausible pairs* (a random entity in the wrong place), not to compare the two relations of an analogy. Training on cross-relation negatives gives a model that really compares relations (77 %, AUC 0.86) and still transfers partly to corrupted negatives. Both ANNc models clearly beat the training-free parallelogram rule.

Example (`analogy_kg.predict`, model trained on cross-relation negatives):

```
Paris : France :: Rome : Italy            ->  VALID (p = 0.99)
Paris : France :: Rome : Albert Einstein  ->  VALID (p = 0.53)   # borderline: a known limitation
```

### What changed compared with the original project

The original student project reported **84 % validation accuracy**. Re-reading the code showed three issues, fixed here, so the numbers above are not directly comparable:

1. **Training loop** — `optimizer.zero_grad()` was never called, so gradients accumulated across all batches.
2. **Data leakage** — analogies were split randomly, so the same entity pairs appeared in training and test analogies. Triples are now split *before* building analogies.
3. **Model selection on the test set** — the reported score was the best epoch on the test split. A separate validation split is now used for early stopping; the test split is used once.

Two other fixes: the inference function embedded the dictionary *keys* (`"Entity1"`…`"Entity4"`) instead of the entities, so it returned the same answer for every input; and corrupted triples are now checked to be absent from the graph (a random replacement could re-create a true triple, or leave it unchanged).

## Quick start

Easiest: open [`colab_run.ipynb`](colab_run.ipynb) in Google Colab with a T4 GPU and run all cells.

Locally (GPU recommended for the embedding step):

```bash
git clone https://github.com/AWARKEH1/yago-analogy-classification.git
cd yago-analogy-classification
pip install -r requirements.txt
gdown 1jvneLm3Jg1_tc2gHqvC4K4sNOI_0AZ2m -O data/train.pkl   # YAGO4-19k training triples

python -m analogy_kg.data  --triples data/train.pkl --out data/analogies
python -m analogy_kg.embed --analogies data/analogies --out data/embeddings.npz
python -m analogy_kg.train --data data/analogies --emb data/embeddings.npz
python -m analogy_kg.predict "Paris" "France" "Rome" "Italy"
pytest
```

## Project structure

```
analogy_kg/
├── data.py      # triples -> valid / invalid analogies, leakage-free split
├── embed.py     # one bge-large embedding per unique entity (.npz cache)
├── model.py     # ANNc
├── train.py     # training, early stopping, baseline, cross-evaluation, figures
└── predict.py   # classify a new analogy
tests/           # dataset builder and model tests
colab_run.ipynb  # end-to-end run on a free GPU
```

## Limitations

- Entities are represented only by the embedding of their **name**, not by their position in the graph.
- Entities (not pairs) can appear in several splits: this is a transductive setting.
- Each pair of an analogy is only checked for its predicate; YAGO predicates are broad (e.g. `citation`, `memberOf`), so some "valid" analogies are semantically weak.

## Credits

Supervised project of the **MSc Natural Language Processing, Université de Lorraine** (2023–2024), by
Abbas Awarkeh, Bassem Jaber and Mohammad Al Takach, supervised by Miguel Couceiro and Pierre Monnin.

- Original group repository: [supervisedproject/ANNc-Based-Analogical-Reasoning-in-YAGO4-19k…](https://github.com/supervisedproject/ANNc-Based-Analogical-Reasoning-in-YAGO4-19k-Classification-of-Valid-and-Invalid-Analogies)
- This repository is my reworked version: bug fixes, leakage-free evaluation, harder cross-relation negatives, baseline, tests and documentation.

References: ANNc — Lim, Prade & Richard (2019), *Solving word analogies: a machine learning perspective*; YAGO4-19k — Hubert, Monnin, Brun & Monticolo; embeddings — `BAAI/bge-large-en-v1.5` (the original project used `Alibaba-NLP/gte-large-en-v1.5`, whose custom code no longer runs with recent `transformers`).
