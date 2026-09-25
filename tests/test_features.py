import numpy as np
import pandas as pd
import pytest

from er.features import FEATURE_COLUMNS, PairFeaturizer

RNG = np.random.default_rng(0)
COLS = ["entity_id", "name_norm", "legal_form", "address_norm"]


def records(prefix, n):
    return pd.DataFrame([(f"{prefix}-{i}", f"shop {i % 37} traders", ["", "llc", "ltd pvt"][i % 3],
                          "" if i % 11 == 0 else f"{i % 50} main street, town {i % 7}") for i in range(n)], columns=COLS)


S1, TARGETS = records("S1", 300), records("S2", 900)
PAIRS = pd.DataFrame({"s1_id": RNG.choice(S1.entity_id, 3000), "cand_id": RNG.choice(TARGETS.entity_id, 3000),
                      "score": RNG.random(3000)}).drop_duplicates(["s1_id", "cand_id"])


def test_output_columns_and_dtype():
    out = PairFeaturizer().transform(PAIRS, S1, TARGETS)
    assert list(out.columns) == ["s1_id", "cand_id"] + FEATURE_COLUMNS
    assert (out[FEATURE_COLUMNS].dtypes == "float32").all()
    assert len(out) == len(PAIRS)


@pytest.mark.parametrize("chunk", [1, 7, 64, 10_000])
def test_chunking_does_not_change_features(chunk):
    whole = PairFeaturizer().transform(PAIRS, S1, TARGETS)
    f = PairFeaturizer()
    f.chunk_s1 = chunk
    chunked = f.transform(PAIRS, S1, TARGETS)
    pd.testing.assert_frame_equal(whole, chunked)


def test_rank_and_gap_are_per_s1_even_across_chunks():
    f = PairFeaturizer()
    f.chunk_s1 = 5
    out = f.transform(PAIRS, S1, TARGETS)
    for _, g in out.groupby("s1_id"):
        assert list(g["rank"]) == list(range(len(g)))
        assert g["score_gap"].iloc[0] == 0 and (g["score_gap"] <= 0).all()


def test_empty_pairs():
    out = PairFeaturizer().transform(PAIRS.iloc[:0], S1, TARGETS)
    assert out.empty and list(out.columns) == ["s1_id", "cand_id"] + FEATURE_COLUMNS
