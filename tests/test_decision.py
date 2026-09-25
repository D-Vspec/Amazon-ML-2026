import pandas as pd

from er.decision import decide, tune_threshold
from er.evaluate import F05Evaluator


def test_tune_threshold_can_pick_above_095():
    # True matches score 0.975; the wrong candidate 0.965: only a threshold in (0.965, 0.975] is perfect.
    scored = pd.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-2"], "cand_id": ["S2-1", "S2-9", "S2-2"],
                           "match_proba": [0.975, 0.965, 0.975]})
    truth = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
    threshold, f05 = tune_threshold(scored, truth, F05Evaluator())
    assert threshold == 0.97 and f05 == 1.0


def test_decide_gives_each_candidate_to_one_s1():
    scored = pd.DataFrame({"s1_id": ["S1-1", "S1-2"], "cand_id": ["S2-1", "S2-1"], "match_proba": [0.8, 0.9]})
    assert decide(scored, 0.5)[["s1_id", "cand_id"]].values.tolist() == [["S1-2", "S2-1"]]
