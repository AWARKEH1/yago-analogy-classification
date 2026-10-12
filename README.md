# Analogy classification on the YAGO4-19k knowledge graph

> Can a neural network tell whether **a : b :: c : d** is a valid analogy, using only sentence embeddings of the four entities?
> Triples from the YAGO4-19k knowledge graph are turned into valid and invalid analogies, and a CNN analogy classifier (**ANNc**) is trained on top of `gte-large-en-v1.5` embeddings.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-ANNc-ee4c2c) ![HF](https://img.shields.io/badge/Hugging%20Face-gte--large-yellow)

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
                each entity ──► gte-large-en-v1.5 embedding (1024-d, mean pooling)
                                    │
                [a | b | c | d] 1024×4 "image" ──► ANNc (2 conv layers + linear) ──► P(valid)
```

The **cross-relation** negatives are the interesting ones: each pair is a true fact, so the model cannot succeed by checking whether a pair "exists" — it has to compare the two relations, which is what analogy is about.

## Results

*Held-out test sets of 5,000 analogies (balanced). Run: [`colab_run.ipynb`](colab_run.ipynb).*

| Model | Trained on | Test: corrupted negatives | Test: cross-relation negatives |
|---|---|---|---|
| Parallelogram baseline (no training) | — | _to be filled_ | _to be filled_ |
| ANNc | corrupted negatives | _to be filled_ | _to be filled_ |
| ANNc | cross-relation negatives | _to be filled_ | _to be filled_ |

<!-- figures: results/figures/training_corrupted.png, results/figures/training_cross_relation.png -->

### What changed compared with the original project

The original student project reported **84 % validation accuracy**. Re-reading the code showed three issues, fixed here, so the new numbers are not directly comparable:

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
├── embed.py     # one gte-large embedding per unique entity (.npz cache)
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

References: ANNc — Lim, Prade & Richard (2019), *Solving word analogies: a machine learning perspective*; YAGO4-19k — Hubert, Monnin, Brun & Monticolo; embeddings — `Alibaba-NLP/gte-large-en-v1.5`.
