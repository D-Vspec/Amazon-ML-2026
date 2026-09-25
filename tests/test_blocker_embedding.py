import pandas as pd
import pytest
import torch

from er.blockers.embedding import PAIR_COLUMNS, EmbeddingBlocker

COLS = ["entity_id", "raw_name", "raw_address", "country"]
TARGETS = pd.DataFrame([
    ("S2-1", "परफेक्ट मीडिया", "C 1 A Sector 27, Noida, UP", "India"),       # Devanagari "Perfect Media"
    ("S2-2", "Baner Tap Pvt Ltd", "Flat B-901, Oakwood Hills, Pune, MH", "India"),
    ("S2-3", "லக்ஷ்மி சர்வீசஸ்", "907, Mumbai, MH", "India"),                   # Tamil "Lakshmi Services"
    ("S2-4", "Harris Better", "8008 14 Ave, Brooklyn, NY", "US"),
    ("S2-5", "Patriot Association", "3343 Dug Hill Rd, Huntsville, AL", "US"),
    ("S2-6", "Perfect Media", "C 1 A Sector 27, Noida, UP", "US"),              # same text, other country
], columns=COLS)


def records(rows):
    return pd.DataFrame(rows, columns=COLS)


@pytest.fixture(scope="module")
def blocker():
    return EmbeddingBlocker("cpu").fit(TARGETS)


def test_output_columns(blocker):
    assert list(blocker.query(records([("S1-1", "Harris Better", "Brooklyn", "US")]), 2).columns) == PAIR_COLUMNS


def test_native_script_matches_latin(blocker):
    pairs = blocker.query(records([("S1-1", "Perfect Media", "C 1 A Sector 27, Noida, UP", "India"),
                                   ("S1-2", "Lakshmi Services Private Limited", "907, Mumbai, MH", "India")]), 1)
    assert dict(zip(pairs.s1_id, pairs.cand_id)) == {"S1-1": "S2-1", "S1-2": "S2-3"}


def test_exact_copy_scores_near_one(blocker):
    pairs = blocker.query(records([("S1-1", "Harris Better", "8008 14 Ave, Brooklyn, NY", "US")]), 1)
    assert pairs.iloc[0].cand_id == "S2-4" and pairs.iloc[0].score > 0.99


def test_candidates_only_from_same_country(blocker):
    pairs = blocker.query(records([("S1-1", "Perfect Media", "C 1 A Sector 27, Noida, UP", "India")]), 10)
    assert "S2-6" not in set(pairs.cand_id)
    assert set(pairs.cand_id) <= {"S2-1", "S2-2", "S2-3"}


def test_unknown_country_gets_no_candidates(blocker):
    pairs = blocker.query(records([("S1-1", "Perfect Media", "Paris", "France")]), 5)
    assert pairs.empty and list(pairs.columns) == PAIR_COLUMNS


def test_at_most_k_sorted_and_k_capped_by_country(blocker):
    pairs = blocker.query(records([("S1-1", "Baner Tap", "Pune", "India"), ("S1-2", "Harris", "NY", "US")]), 50)
    assert (pairs.groupby("s1_id").size() == pd.Series({"S1-1": 3, "S1-2": 3})).all()  # 3 targets per country
    for _, g in pairs.groupby("s1_id"):
        assert list(g.score) == sorted(g.score, reverse=True)


def test_score_pairs_matches_query_scores_and_scores_unproposed_pairs(blocker):
    rows = records([("S1-1", "Perfect Media", "C 1 A Sector 27, Noida, UP", "India")])
    pairs = blocker.query(rows, 1)
    assert blocker.score_pairs(pairs) == pytest.approx(pairs.score.to_numpy(), abs=1e-4)
    extra = pd.DataFrame({"s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-2", "S2-3"]})  # not in its top 1
    scores = blocker.score_pairs(extra)
    assert (scores > 0).all() and (scores < pairs.score.iloc[0]).all()


def test_e5_prefix_only_for_e5_models(blocker):
    assert blocker.prefix == "query: "


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cpu_and_gpu_agree(blocker):
    rows = records([("S1-1", "Perfect Media", "C 1 A Sector 27, Noida, UP", "India"),
                    ("S1-2", "Patriot Assoc", "Dug Hill Road, Huntsville", "US")])
    cpu = blocker.query(rows, 3)
    gpu = EmbeddingBlocker("cuda").fit(TARGETS).query(rows, 3)
    assert list(cpu.cand_id) == list(gpu.cand_id)
    assert cpu.score.to_numpy() == pytest.approx(gpu.score.to_numpy(), abs=5e-3)  # GPU runs in fp16
