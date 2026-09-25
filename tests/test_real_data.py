"""Checks on the real challenge data. Skipped when the (git-ignored) dataset is not present."""

import csv
import itertools
import random
import re
from pathlib import Path

import pandas as pd
import pytest

from er.normalize import RuleNormalizer
from er.transliterate import AnyAsciiTransliterator

DATA = Path(__file__).parent.parent / "6ab10eb3b23ba_student_resource/student_resource/dataset"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="challenge dataset not present")

t, n = AnyAsciiTransliterator(), RuleNormalizer()


def _pipeline(df):
    return n.transform(t.transform(df))


@pytest.fixture(scope="module")
def sample():
    frames = [pd.read_csv(DATA / sp / f"{sp}_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False,
                          quoting=csv.QUOTE_NONE, nrows=50_000)
              for sp in ("train", "test") for s in (1, 2, 3)]
    return _pipeline(pd.concat(frames, ignore_index=True))


def test_all_countries_present(sample):
    assert {"US", "India", "France"} <= set(sample.country)


@pytest.mark.parametrize("col", ["name_norm", "legal_form", "address_norm"])
def test_output_ascii_and_clean_spacing(sample, col):
    v = sample[col]
    assert v.map(str.isascii).all()
    assert not v.str.contains("  ").any()
    assert (v == v.str.strip()).all()


def test_names_idempotent(sample):
    again = [n.normalize_name(x, c)[0] for x, c in zip(sample.name_norm, sample.country)]
    assert again == sample.name_norm.tolist()


def test_addresses_idempotent(sample):
    again = [n.normalize_address(x, c) for x, c in zip(sample.address_norm, sample.country)]
    assert again == sample.address_norm.tolist()


def test_state_codes_survive(sample):
    raw_last = sample.business_address.str.split(",").str[-1].str.strip()
    norm_last = sample.address_norm.str.split(", ").str[-1]
    for code in ("CT", "FL"):
        assert (norm_last[raw_last == code] == code.lower()).all()


@pytest.fixture(scope="module")
def ground_truth_pairs():
    """Matched pairs for 3000 S1 entities plus as many random same-country non-matches."""
    rng = random.Random(0)
    with open(DATA / "train/train_ground_truth.tsv") as f:
        next(f)
        gt = {i: m.split(",") for i, _, m in (l.rstrip("\n").partition("\t") for l in itertools.islice(f, 3000)) if m}
    need = set(gt) | {x for v in gt.values() for x in v}
    rows = []
    for s in (1, 2, 3):
        with open(DATA / f"train/train_source{s}.tsv") as f:
            next(f)
            rows += [p[:4] for p in (l.rstrip("\n").split("\t") for l in f) if p[0] in need]
    recs = _pipeline(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))
    recs = recs.set_index("entity_id")
    pos = [(a, b) for a, v in gt.items() for b in v if a in recs.index and b in recs.index]
    s1, other = [a for a in gt if a in recs.index], [x for x in recs.index if not x.startswith("S1-")]
    neg = []
    while len(neg) < len(pos):
        a, b = rng.choice(s1), rng.choice(other)
        if b not in gt[a] and recs.country[a] == recs.country[b]:
            neg.append((a, b))
    return recs, pos, neg


def _exact_rate(recs, pairs, col):
    return sum(recs[col][a] == recs[col][b] != "" for a, b in pairs) / len(pairs)


def _baseline(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def test_normalization_makes_matches_agree_more(ground_truth_pairs):
    recs, pos, _ = ground_truth_pairs
    recs = recs.assign(name_base=recs.business_name.map(_baseline), addr_base=recs.business_address.map(_baseline))
    # Measured on 20k entities: names 26% -> 52%, addresses 8% -> 28% exactly equal.
    assert _exact_rate(recs, pos, "name_norm") > 1.5 * _exact_rate(recs, pos, "name_base")
    assert _exact_rate(recs, pos, "address_norm") > 2 * _exact_rate(recs, pos, "addr_base")


def test_non_matches_do_not_collide(ground_truth_pairs):
    recs, _, neg = ground_truth_pairs
    assert _exact_rate(recs, neg, "name_norm") < 0.001
    assert _exact_rate(recs, neg, "address_norm") < 0.001
