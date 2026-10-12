"""Tests for the analogy dataset builder (pure Python, no GPU needed)."""

import random

import pandas as pd
import pytest

from analogy_kg.data import (build_analogies, corrupt, entity_to_text, load_triples,
                             split_triples, uri_tail)

Y = "http://yago-knowledge.org/resource/"
S = "http://schema.org/"


@pytest.fixture
def triples():
    rng = random.Random(0)
    people = [f"Person_{i}" for i in range(30)]
    cities = [f"City_{i}" for i in range(10)]
    jobs = [f"Job_{i}" for i in range(8)]
    t = {(p, "birthPlace", rng.choice(cities)) for p in people}
    t |= {(p, "hasOccupation", rng.choice(jobs)) for p in people}
    return sorted(t)


def test_uri_and_text_cleaning():
    assert uri_tail(f"<{Y}Polaris_(2005_role-playing_game)>") == "Polaris_(2005_role-playing_game)"
    assert entity_to_text("Anna_of_Masovia_Q527518") == "Anna of Masovia"
    assert entity_to_text("_Q527518") == ""


def test_load_triples(tmp_path):
    raw = [(f"<{Y}Paris>", f"<{S}containedInPlace>", f"<{Y}France>"),
           (f"<{Y}_Q1>", f"<{S}knows>", f"<{Y}Bob>")]  # no readable name -> dropped
    path = tmp_path / "train.pkl"
    pd.to_pickle(raw, path)
    assert load_triples(path) == [("Paris", "containedInPlace", "France")]


def test_split_is_disjoint(triples):
    splits = split_triples(triples)
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not (train & test) and not (train & val) and not (val & test)
    assert len(train | val | test) == len(triples)


def test_corrupt_never_returns_a_known_triple(triples):
    known = set(triples)
    entities = sorted({t[0] for t in triples} | {t[2] for t in triples})
    rng = random.Random(1)
    for t in triples:
        c = corrupt(t, entities, known, rng)
        assert c not in known and c[1] == t[1]
        assert (c[0] == t[0]) != (c[2] == t[2])  # exactly one side changed


@pytest.mark.parametrize("negatives", ["corrupted", "cross_relation"])
def test_build_analogies(triples, negatives):
    known = set(triples)
    entities = sorted({t[0] for t in triples} | {t[2] for t in triples})
    df = build_analogies(triples, 200, entities, known, negatives=negatives)
    assert len(df) == 200 and df["label"].mean() == 0.5
    pos = df[df.label == 1]
    for row in pos.itertuples():
        assert (row.a, row.relation, row.b) in known
        assert (row.c, row.relation, row.d) in known
    neg = df[df.label == 0]
    if negatives == "cross_relation":
        assert all(p != q for p, q in neg.relation.str.split("|"))
    else:
        for row in neg.itertuples():
            assert (row.a, row.relation, row.b) not in known


def test_no_pair_leaks_from_test_to_train(triples):
    known = set(triples)
    entities = sorted({t[0] for t in triples} | {t[2] for t in triples})
    splits = split_triples(triples, val=0.2, test=0.2)
    train = build_analogies(splits["train"], 300, entities, known)
    test = build_analogies(splits["test"], 100, entities, known)
    pos_pairs = lambda df: ({(r.a, r.b) for r in df[df.label == 1].itertuples()}
                            | {(r.c, r.d) for r in df[df.label == 1].itertuples()})
    assert not pos_pairs(train) & pos_pairs(test)
