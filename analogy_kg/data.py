"""Turn YAGO4-19k triples into a balanced dataset of valid / invalid analogies.

A valid analogy a:b::c:d is built from two *true* triples sharing the same
predicate: (a, p, b) and (c, p, d).

Two kinds of invalid analogies are supported:

* ``corrupted``      - (original project) both pairs come from corrupted triples:
                       the head or the tail is replaced by a random entity so that
                       the resulting triple does NOT exist in the knowledge graph.
* ``cross_relation`` - harder negatives: both pairs are TRUE facts, but they come
                       from two different predicates (a, p, b) and (c, q, d), p != q.
                       A model can only solve these by comparing the two relations,
                       not by checking whether each pair is a known fact.

To avoid data leakage, triples are split into train / validation / test sets
*before* analogies are built, so no entity pair of the test set is seen in training.

Usage:
    python -m analogy_kg.data --triples data/train.pkl --out data/analogies
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

Triple = tuple[str, str, str]


# --- Loading and cleaning ------------------------------------------------------

def uri_tail(uri: str) -> str:
    """'<http://yago-knowledge.org/resource/Paris>' -> 'Paris'."""
    return str(uri).strip().strip("<>").rstrip("/").split("/")[-1]


def entity_to_text(entity: str) -> str:
    """Readable text for the embedding model: underscores -> spaces, drop
    trailing Wikidata ids ('Anna_of_Masovia_Q123' -> 'Anna of Masovia')."""
    text = entity.replace("_", " ")
    text = re.sub(r"\s*Q\d+$", "", text)
    return text.strip()


def load_triples(path: Path) -> list[Triple]:
    """Load the YAGO4-19k ``train.pkl`` file (an iterable of (s, p, o) URIs)."""
    raw = pd.read_pickle(path)
    if isinstance(raw, pd.DataFrame):
        raw = raw.itertuples(index=False, name=None)
    triples = []
    for s, p, o in raw:
        s, p, o = uri_tail(s), uri_tail(p), uri_tail(o)
        # keep only triples whose entities still have a readable name
        if entity_to_text(s) and entity_to_text(o):
            triples.append((s, p, o))
    return sorted(set(triples))


def split_triples(
    triples: list[Triple], val: float = 0.1, test: float = 0.1, seed: int = 42
) -> dict[str, list[Triple]]:
    """Random split of the TRIPLES (not of the analogies)."""
    rng = random.Random(seed)
    shuffled = triples[:]
    rng.shuffle(shuffled)
    n_test, n_val = int(len(shuffled) * test), int(len(shuffled) * val)
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


# --- Negative sampling ---------------------------------------------------------

def corrupt(triple: Triple, entities: list[str], known: set[Triple],
            rng: random.Random, max_tries: int = 100) -> Triple:
    """Replace the head or the tail by a random entity, making sure the new
    triple is different from the original and absent from the whole graph."""
    s, p, o = triple
    for _ in range(max_tries):
        e = rng.choice(entities)
        candidate = (e, p, o) if rng.random() < 0.5 else (s, p, e)
        if candidate != triple and candidate not in known:
            return candidate
    raise RuntimeError(f"Could not corrupt {triple}")


# --- Analogy generation --------------------------------------------------------

def _by_predicate(triples: list[Triple]) -> dict[str, list[Triple]]:
    groups: dict[str, list[Triple]] = defaultdict(list)
    for t in triples:
        groups[t[1]].append(t)
    # an analogy needs at least two triples with the same predicate
    return {p: ts for p, ts in groups.items() if len(ts) >= 2}


def build_analogies(
    triples: list[Triple],
    n_samples: int,
    entities: list[str],
    known: set[Triple],
    negatives: str = "corrupted",
    seed: int = 42,
) -> pd.DataFrame:
    """Sample ``n_samples`` analogies (50 % valid, 50 % invalid) from ``triples``."""
    if negatives not in {"corrupted", "cross_relation"}:
        raise ValueError("negatives must be 'corrupted' or 'cross_relation'")
    rng = random.Random(seed)
    groups = _by_predicate(triples)
    predicates = list(groups)
    if negatives == "cross_relation" and len(predicates) < 2:
        raise ValueError("cross_relation negatives need at least two predicates")
    # sample predicates proportionally to their number of triples
    weights = [len(groups[p]) for p in predicates]

    rows = []
    for i in range(n_samples):
        p = rng.choices(predicates, weights=weights)[0]
        t1, t2 = rng.sample(groups[p], 2)
        if i % 2 == 0:
            label, relation = 1, p
        else:
            label = 0
            if negatives == "corrupted":
                t1 = corrupt(t1, entities, known, rng)
                t2 = corrupt(t2, entities, known, rng)
                relation = p
            else:
                q = p
                while q == p:
                    q = rng.choices(predicates, weights=weights)[0]
                t2 = rng.choice(groups[q])
                relation = f"{p}|{q}"
        rows.append({
            "a": t1[0], "b": t1[2], "c": t2[0], "d": t2[2],
            "relation": relation, "label": label,
        })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triples", type=Path, default=Path("data/train.pkl"))
    parser.add_argument("--out", type=Path, default=Path("data/analogies"))
    parser.add_argument("--n-train", type=int, default=40_000)
    parser.add_argument("--n-eval", type=int, default=5_000,
                        help="number of analogies for validation and for test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    triples = load_triples(args.triples)
    known = set(triples)
    entities = sorted({t[0] for t in triples} | {t[2] for t in triples})
    splits = split_triples(triples, seed=args.seed)
    print(f"{len(triples)} triples, {len(entities)} entities, "
          f"{len({t[1] for t in triples})} predicates")
    print({k: len(v) for k, v in splits.items()})

    args.out.mkdir(parents=True, exist_ok=True)
    sizes = {"train": args.n_train, "val": args.n_eval, "test": args.n_eval}
    for negatives in ("corrupted", "cross_relation"):
        for split, n in sizes.items():
            df = build_analogies(splits[split], n, entities, known,
                                 negatives=negatives, seed=args.seed)
            path = args.out / f"{split}_{negatives}.csv"
            df.to_csv(path, index=False)
            print(f"{path}: {len(df)} analogies")


if __name__ == "__main__":
    main()
