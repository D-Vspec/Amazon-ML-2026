import csv
from pathlib import Path

import pandas as pd
import pytest

from er.evaluate import F05Evaluator
from er.split import HashSplitter

sp = HashSplitter()
TRAIN = Path(__file__).parent.parent / "6ab10eb3b23ba_student_resource/student_resource/dataset/train"


def _ids(path):
    return set(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE).iloc[:, 0])


def test_set_of_deterministic_and_in_range():
    ids = [f"S1-{i}" for i in range(5000)]
    assert [sp.set_of(x) for x in ids] == [sp.set_of(x) for x in ids]
    assert {sp.set_of(x) for x in ids} == set(range(10))


@pytest.mark.parametrize("n", [2, 5, 20])
def test_other_set_counts(n):
    assert {HashSplitter(n).set_of(f"S1-{i}") for i in range(5000)} == set(range(n))


def test_sets_roughly_equal():
    counts = pd.Series([sp.set_of(f"S1-{i}") for i in range(100_000)]).value_counts()
    assert counts.min() > 9_500 and counts.max() < 10_500


def _write_fixture(d: Path):
    s1 = [f"S1-{i}" for i in range(200)]
    gt = {s: ([f"S2-m{i}", f"S3-m{i}"] if i % 5 else []) for i, s in enumerate(s1)}  # every 5th is a singleton
    distractors = [f"S2-d{i}" for i in range(300)]
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    rows = lambda ids: header + "".join(f"{x}\tn\ta\tUS\n" for x in ids)
    (d / "train_source1.tsv").write_text(rows(s1))
    (d / "train_source2.tsv").write_text(rows([m[0] for m in gt.values() if m] + distractors))
    (d / "train_source3.tsv").write_text(rows([m[1] for m in gt.values() if m]))
    (d / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n" +
                                              "".join(f"{s}\t{','.join(m)}\n" for s, m in gt.items()))
    return gt, distractors


@pytest.fixture
def written(tmp_path):
    gt, distractors = _write_fixture(tmp_path)
    counts = sp.write(tmp_path, tmp_path / "out")
    return tmp_path / "out", gt, distractors, counts


def test_every_record_in_exactly_one_set(written):
    out, gt, distractors, _ = written
    for name, expected in [("source1.tsv", set(gt)),
                           ("source2.tsv", {m[0] for m in gt.values() if m} | set(distractors)),
                           ("source3.tsv", {m[1] for m in gt.values() if m})]:
        per_set = [_ids(out / f"set_{k}" / name) for k in range(10)]
        assert sum(len(s) for s in per_set) == len(expected)
        assert set().union(*per_set) == expected


def test_matches_and_ground_truth_follow_their_s1(written):
    out, gt, _, _ = written
    for k in range(10):
        d = out / f"set_{k}"
        truth = F05Evaluator.load(d / "ground_truth.tsv")
        present = _ids(d / "source2.tsv") | _ids(d / "source3.tsv")
        assert set(truth) == _ids(d / "source1.tsv")
        assert all(ids <= present for ids in truth.values())
        assert all(sp.set_of(s1) == k for s1 in truth)


def test_counts_match_files(written):
    out, _, _, counts = written
    for k in range(10):
        for name, n in counts[k].items():
            assert n == len(_ids(out / f"set_{k}" / name))


# ---------------------------------------------------------------- real data (skipped without the dataset)

@pytest.mark.skipif(not TRAIN.exists(), reason="challenge dataset not present")
def test_real_sets_are_self_contained_and_representative(tmp_path):
    counts = sp.write(TRAIN, tmp_path)
    total_s1 = sum(c["source1.tsv"] for c in counts.values())
    for k in (0, 7):
        d = tmp_path / f"set_{k}"
        truth = F05Evaluator.load(d / "ground_truth.tsv")
        present = _ids(d / "source2.tsv") | _ids(d / "source3.tsv")
        assert all(ids <= present for ids in truth.values())  # every true match is in the set
        n23 = counts[k]["source2.tsv"] + counts[k]["source3.tsv"]
        matched = sum(len(v) for v in truth.values())
        # Full-data values: 5.59% singletons, 26.0% distractors.
        assert abs(sum(not v for v in truth.values()) / len(truth) - 0.0559) < 0.01
        assert abs((n23 - matched) / n23 - 0.260) < 0.01
        assert abs(len(truth) / total_s1 - 0.10) < 0.005
