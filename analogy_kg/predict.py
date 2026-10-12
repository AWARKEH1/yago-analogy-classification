"""Classify a new analogy a : b :: c : d with a trained ANNc model.

Usage:
    python -m analogy_kg.predict "Paris" "France" "Rome" "Italy"
    python -m analogy_kg.predict "Paris" "France" "Rome" "Italy" --model results/annc_cross_relation.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import entity_to_text
from .embed import Embedder
from .model import ANNc


def classify(entities: list[str], model: ANNc, embedder: Embedder) -> float:
    """Return the probability that the analogy is valid."""
    # Embed the four entity VALUES (the original inference embedded the dict keys
    # "Entity1".."Entity4", so it returned the same answer for every input).
    texts = [entity_to_text(e) for e in entities]
    vectors = torch.from_numpy(embedder.encode(texts))    # [4, emb]
    x = vectors.T.unsqueeze(0).to(embedder.device)         # [1, emb, 4]
    model.eval()
    with torch.no_grad():
        return float(model.predict_proba(x)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entities", nargs=4, metavar=("A", "B", "C", "D"))
    parser.add_argument("--model", type=Path, default=Path("results/annc_cross_relation.pt"))
    args = parser.parse_args()

    embedder = Embedder()
    model = ANNc().to(embedder.device)
    model.load_state_dict(torch.load(args.model, map_location=embedder.device))
    p = classify(args.entities, model, embedder)
    a, b, c, d = args.entities
    print(f"{a} : {b} :: {c} : {d}  ->  {'VALID' if p >= 0.5 else 'INVALID'} (p = {p:.2f})")


if __name__ == "__main__":
    main()
