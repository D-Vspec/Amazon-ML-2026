"""Label blocking candidates against the ground truth to build the matcher's training set.

Code: src/er/training_pairs.py. See docs/evaluation.md (F05Evaluator.load: {s1_id: set(ids)}) and
docs/blocking.md (candidate contract: s1_id, cand_id, score).

Positives: candidate pairs that are in the ground truth.
Negatives: candidate pairs the blocker returned for that S1 that are NOT in the ground truth. These are
hard negatives by construction: the blocker already scored them as similar, which is exactly what the
matcher has to learn to tell apart from a real match.

A true match the blocker missed can never appear here as a positive; that ceiling is what
F05Evaluator.blocking_recall measures, and no amount of matcher training recovers it.
"""

import pandas as pd

IdLists = dict[str, set[str]]


def build_training_pairs(candidates: pd.DataFrame, ground_truth: IdLists) -> pd.DataFrame:
    """candidates: s1_id, cand_id, [score, ...]. Returns the same frame with a 0/1 `label` column."""
    out = candidates.copy()
    out["label"] = [
        int(cand in ground_truth.get(s1, ())) for s1, cand in zip(out["s1_id"], out["cand_id"])
    ]
    return out


def summarize(labeled: pd.DataFrame, ground_truth: IdLists) -> dict:
    """Quick sanity check before training: class balance, and how many true matches never showed up
    as a candidate at all (a blocking miss, invisible to this labeling step)."""
    n_pos, n_neg = int(labeled["label"].sum()), int((labeled["label"] == 0).sum())
    total_true = sum(len(ids) for ids in ground_truth.values())
    return {
        "positives": n_pos,
        "negatives": n_neg,
        "negative_rate": n_neg / (n_pos + n_neg) if (n_pos + n_neg) else 0.0,
        "true_matches_total": total_true,
        "true_matches_recovered": n_pos,          # equals blocking_recall * total_true
        "true_matches_missed_by_blocking": total_true - n_pos,
    }