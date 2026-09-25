import pandas as pd
import pytest
import torch

from er.blockers.tfidf import PAIR_COLUMNS, TfidfNgramBlocker

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def records(rows):
    return pd.DataFrame(rows, columns=["entity_id", "name_norm", "address_norm", "country"])


TARGETS = records([
    ("S2-1", "agro india", "office 203 mumbai mh", "India"),
    ("S3-1", "agro indai", "office 203 mumbai mh", "India"),          # typo
    ("S2-2", "baner tap", "flat b 901 baner pune mh", "India"),
    ("S2-3", "clyial pony", "9285 perseverance drive harrisburg nc", "US"),
    ("S3-3", "pony clyial", "9285 perseverance drive harrisburg nc", "US"),  # reordered
    ("S2-4", "harris better", "8008 14 avenue brooklyn ny", "US"),
    ("S2-5", "patriot association", "3343 dug hill road huntsville al", "US"),
    ("S2-6", "agro india", "office 203 mumbai mh", "US"),             # same text, other country
])


@pytest.fixture(params=DEVICES)
def blocker(request):
    return TfidfNgramBlocker(device=request.param, max_df=1.0).fit(TARGETS)  # max_df off: tiny corpus


def query(blocker, rows, k=3):
    return blocker.query(records(rows), k)


def test_output_columns(blocker):
    assert list(query(blocker, [("S1-1", "agro india", "office 203 mumbai mh", "India")]).columns) == PAIR_COLUMNS


def test_exact_copy_ranks_first(blocker):
    pairs = query(blocker, [("S1-1", "harris better", "8008 14 avenue brooklyn ny", "US")])
    assert pairs.iloc[0].cand_id == "S2-4"
    assert pairs.iloc[0].score == pytest.approx(1.0, abs=1e-5)


def test_typo_and_reordering_are_found(blocker):
    pairs = query(blocker, [("S1-1", "agro india", "office 203 mumbai mh", "India"),
                            ("S1-2", "clyial pony", "9285 perseverance dr harrisburg nc", "US")])
    assert {"S2-1", "S3-1"} <= set(pairs[pairs.s1_id == "S1-1"].cand_id)
    assert {"S2-3", "S3-3"} <= set(pairs[pairs.s1_id == "S1-2"].cand_id)


def test_candidates_only_from_same_country(blocker):
    pairs = query(blocker, [("S1-1", "agro india", "office 203 mumbai mh", "India")], k=10)
    assert "S2-6" not in set(pairs.cand_id)  # identical text, but US
    country = dict(zip(TARGETS.entity_id, TARGETS.country))
    assert {country[c] for c in pairs.cand_id} == {"India"}


def test_unknown_country_gets_no_candidates(blocker):
    pairs = query(blocker, [("S1-1", "agro india", "office 203 mumbai mh", "France")])
    assert pairs.empty and list(pairs.columns) == PAIR_COLUMNS


def test_at_most_k_per_s1_sorted_by_score(blocker):
    pairs = query(blocker, [("S1-1", "agro india", "office 203 mumbai mh", "India"),
                            ("S1-2", "harris better", "8008 14 avenue brooklyn ny", "US")], k=2)
    for _, group in pairs.groupby("s1_id"):
        assert len(group) <= 2
        assert list(group.score) == sorted(group.score, reverse=True)


def test_k_larger_than_country(blocker):
    pairs = query(blocker, [("S1-1", "agro india", "office 203 mumbai mh", "India")], k=50)
    assert len(pairs) <= 3  # only 3 Indian targets


def test_nothing_in_common_gives_no_pairs(blocker):
    assert query(blocker, [("S1-1", "zzzzqqq", "xxxxwww", "US")]).empty


def test_scores_in_unit_interval(blocker):
    pairs = query(blocker, [("S1-1", "patriot assoc", "dug hill road huntsville al", "US")], k=5)
    assert ((pairs.score > 0) & (pairs.score <= 1.0 + 1e-5)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cpu_and_gpu_agree():
    rows = [("S1-1", "agro india", "office 203 mumbai mh", "India"),
            ("S1-2", "clyial pony", "9285 perseverance dr harrisburg nc", "US")]
    cpu = TfidfNgramBlocker("cpu", max_df=1.0).fit(TARGETS).query(records(rows), 3)
    gpu = TfidfNgramBlocker("cuda", max_df=1.0).fit(TARGETS).query(records(rows), 3)
    assert list(cpu.cand_id) == list(gpu.cand_id)
    assert cpu.score.to_numpy() == pytest.approx(gpu.score.to_numpy(), abs=1e-5)


def test_cuda_falls_back_to_cpu_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert TfidfNgramBlocker("cuda").device.type == "cpu"
