import pytest

from er.evaluate import F05Evaluator

e = F05Evaluator()


def test_readme_worked_example():
    # README: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> P=2/3, R=1 -> 0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert e.entity_score(pred, truth) == pytest.approx(0.7142857)


@pytest.mark.parametrize("pred, truth, expected", [
    (set(), set(), 1.0),                        # singleton, predicted empty
    ({"S2-1"}, set(), 0.0),                     # singleton, any prediction
    ({"S2-1"}, {"S2-1"}, 1.0),                  # perfect
    (set(), {"S2-1"}, 0.0),                     # missed everything
    ({"S2-9"}, {"S2-1"}, 0.0),                  # all wrong
    ({"S2-1"}, {"S2-1", "S3-2"}, 1.25 * 0.5 / (0.25 + 0.5)),  # P=1, R=0.5 -> 0.833
    ({"S2-1", "S2-2"}, {"S2-1"}, 1.25 * 0.5 / (0.125 + 1)),   # P=0.5, R=1 -> 0.556
])
def test_entity_score(pred, truth, expected):
    assert e.entity_score(pred, truth) == pytest.approx(expected)


def test_precision_weighted_more_than_recall():
    truth = {"a", "b", "c", "d"}
    half_recall = e.entity_score({"a", "b"}, truth)                   # P=1, R=0.5
    half_precision = e.entity_score({"a", "b", "c", "d", "x", "y", "z", "w"}, truth)  # P=0.5, R=1
    assert half_recall > half_precision


def test_macro_average_over_truth_entities():
    truth = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1"}}
    pred = {"S1-1": {"S2-1"}, "S1-2": {"S2-5"}, "S1-3": set()}
    assert e.score(pred, truth) == pytest.approx((1 + 0 + 0) / 3)


def test_missing_s1_is_empty_prediction():
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    assert e.score({}, truth) == pytest.approx(0.5)  # non-singleton 0, singleton 1


def test_extra_s1_in_pred_ignored():
    truth = {"S1-1": {"S2-1"}}
    assert e.score({"S1-1": {"S2-1"}, "S1-9": {"S2-7"}}, truth) == 1.0


def test_perfect_and_empty_baselines():
    truth = {f"S1-{i}": ({f"S2-{i}"} if i % 4 else set()) for i in range(100)}
    assert e.score(truth, truth) == 1.0
    assert e.score({}, truth) == pytest.approx(0.25)  # = singleton rate


def test_blocking_recall_and_mean_candidates():
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}, "S1-3": set()}
    cands = {"S1-1": {"S2-1", "S2-9"}, "S1-2": {"S2-2", "S3-8", "S3-9"}}
    assert e.blocking_recall(cands, truth) == pytest.approx(2 / 3)
    assert e.mean_candidates(cands, truth) == pytest.approx(5 / 3)


def test_blocking_recall_all_singletons():
    assert e.blocking_recall({}, {"S1-1": set()}) == 1.0


def test_load(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\t\nS1-3\tS2-3,S2-3\n")
    assert e.load(p) == {"S1-1": {"S2-1", "S3-2"}, "S1-2": set(), "S1-3": {"S2-3"}}
