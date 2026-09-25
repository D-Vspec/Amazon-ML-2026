import csv
from pathlib import Path

import pandas as pd
import pytest

from er.evaluate import F05Evaluator
from er.split import HashSplitter

sp = HashSplitter()
TRAIN = Path(__file__).parent.parent / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"


def test_bucket_deterministic_and_in_range():
    ids = [f"S1-{i}" for i in range(5000)]
    assert [sp.bucket(x) for x in ids] == [sp.bucket(x) for x in ids]
    assert all(0 <= sp.bucket(x) < 100 for x in ids)


def test_buckets_roughly_uniform():
    counts = pd.Series([sp.bucket(f"S1-{i}") for i in range(100_000)]).value_counts()
    assert len(counts) == 100
    assert counts.min() > 850 and counts.max() < 1150  # ~1000 each


def test_tiers_nested_and_train_disjoint_from_val():
    t = {k: set(v) for k, v in sp.tier_buckets.items()}
    assert t["val_1"] < t["val_5"] < t["val_10"]
    assert t["train_10"] < t["train"]
    assert not t["val_10"] & t["train"]
    assert t["val_10"] | t["train"] == set(range(100))


def _write_fixture(d: Path):
    """Two clusters, a singleton and distractors, with ids chosen to land in known buckets."""
    by_bucket = {}
    i = 0
    while len(by_bucket) < 100:
        by_bucket.setdefault(sp.bucket(f"S1-{i}"), f"S1-{i}")
        i += 1
    s1_val, s1_train, s1_single = by_bucket[0], by_bucket[50], by_bucket[3]
    distractors = [f"S2-d{i}" for i in range(400)]
    gt = {s1_val: ["S2-a", "S3-a"], s1_train: ["S2-b"], s1_single: []}
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (d / "train_source1.tsv").write_text(header + "".join(f"{s}\tn\ta\tUS\n" for s in gt))
    (d / "train_source2.tsv").write_text(header + "".join(f"{x}\tn\ta\tUS\n" for x in ["S2-a", "S2-b", *distractors]))
    (d / "train_source3.tsv").write_text(header + "S3-a\tn\ta\tUS\n")
    (d / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" +
                                              "".join(f"{s}\t{','.join(m)}\n" for s, m in gt.items()))
    return gt, distractors


def _ids(path):
    return set(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).iloc[:, 0])


def test_write_keeps_clusters_together(tmp_path):
    gt, distractors = _write_fixture(tmp_path)
    s1_val, s1_train, s1_single = gt
    counts = sp.write(tmp_path, tmp_path / "out")
    out = tmp_path / "out"
    # val cluster (bucket 0) is in every val tier, with both matches, and never in train
    for t in ("val_1", "val_5", "val_10"):
        assert s1_val in _ids(out / t / "source1.tsv")
        assert {"S2-a"} <= _ids(out / t / "source2.tsv")
        assert "S3-a" in _ids(out / t / "source3.tsv")
    assert s1_val not in _ids(out / "train" / "source1.tsv")
    assert "S2-a" not in _ids(out / "train" / "source2.tsv")
    # train cluster (bucket 50) only in train
    assert s1_train in _ids(out / "train" / "source1.tsv") and "S2-b" in _ids(out / "train" / "source2.tsv")
    assert s1_train not in _ids(out / "val_10" / "source1.tsv")
    # singleton (bucket 3) is in val_5 but not val_1, and its GT row comes with it
    assert s1_single in _ids(out / "val_5" / "ground_truth.tsv")
    assert s1_single not in _ids(out / "val_1" / "ground_truth.tsv")
    # every distractor lands in exactly one of val_10 / train
    val, train = _ids(out / "val_10" / "source2.tsv"), _ids(out / "train" / "source2.tsv")
    assert all((d in val) != (d in train) for d in distractors)
    assert counts["val_10"]["source2.tsv"] + counts["train"]["source2.tsv"] == 2 + len(distractors)


def test_written_ground_truth_is_consistent(tmp_path):
    _write_fixture(tmp_path)
    sp.write(tmp_path, tmp_path / "out")
    for t in sp.tier_buckets:
        d = tmp_path / "out" / t
        gt = F05Evaluator.load(d / "ground_truth.tsv")
        present = _ids(d / "source2.tsv") | _ids(d / "source3.tsv")
        assert set(gt) == _ids(d / "source1.tsv")
        assert all(ids <= present for ids in gt.values())


# ---------------------------------------------------------------- real data (skipped without the dataset)

real = pytest.mark.skipif(not TRAIN.exists(), reason="challenge dataset not present")


@pytest.fixture(scope="module")
def real_splits(tmp_path_factory):
    out = tmp_path_factory.mktemp("splits")
    counts = sp.write(TRAIN, out, tiers=["val_1", "val_5"])
    return out, counts


@real
def test_real_val_tier_is_self_contained(real_splits):
    out, _ = real_splits
    d = out / "val_5"
    gt = F05Evaluator.load(d / "ground_truth.tsv")
    present = _ids(d / "source2.tsv") | _ids(d / "source3.tsv")
    assert set(gt) == _ids(d / "source1.tsv")
    assert all(ids <= present for ids in gt.values())  # every true match is in the subset


@real
def test_real_val_tier_is_representative(real_splits):
    out, counts = real_splits
    d = out / "val_5"
    gt = F05Evaluator.load(d / "ground_truth.tsv")
    s1 = pd.read_csv(d / "source1.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    n_s23 = counts["val_5"]["source2.tsv"] + counts["val_5"]["source3.tsv"]
    matched = sum(len(v) for v in gt.values())
    # Full-data values measured on train: 5.6% singletons, 26.0% distractors, ~5% of S1.
    assert abs(sum(not v for v in gt.values()) / len(gt) - 0.056) < 0.01
    assert abs((n_s23 - matched) / n_s23 - 0.260) < 0.01
    assert abs(len(gt) / 2_206_821 - 0.05) < 0.002
    assert set(s1.country) == {"US", "India"}
