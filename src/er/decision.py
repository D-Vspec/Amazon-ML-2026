"""Turn scored candidate pairs into the final match decision.

Code: src/er/decision.py. See docs/matching.md and docs/evaluation.md (F05Evaluator).

Three steps, in order:
1. Threshold: drop pairs below `threshold`. Singletons fall out naturally here (every candidate for
   that S1 scores low, so nothing survives) — that's what earns the full 1.0 (docs/evaluation.md).
2. Margin (optional): within one S1's surviving candidates, drop any more than `margin` below the
   best score for that S1. Catches cases where the top candidate is clearly right but a second,
   weaker one only barely cleared the threshold.
3. Dedup: every S2/S3 record belongs to at most one S1 in the training ground truth, 0 exceptions
   (docs/data_splits.md). If a candidate survives for more than one S1, keep it only for the S1 that
   scored it highest — free precision, since a duplicate assignment can only be wrong at least once.
"""

import pandas as pd

Truth = dict[str, set[str]]


def decide(scored: pd.DataFrame, threshold: float, margin: float | None = None) -> pd.DataFrame:
    """scored: s1_id, cand_id, match_proba, ... (matcher output). Returns the filtered final matches."""
    df = scored[scored["match_proba"] >= threshold]
    if margin is not None and len(df):
        best = df.groupby("s1_id")["match_proba"].transform("max")
        df = df[df["match_proba"] >= best - margin]
    df = df.sort_values("match_proba", ascending=False).drop_duplicates(subset="cand_id", keep="first")
    return df.reset_index(drop=True)


def tune_threshold(scored: pd.DataFrame, truth: Truth, evaluator, thresholds=None,
                   margin: float | None = None) -> tuple[float, float]:
    """Grid-search the threshold that maximizes macro F0.5 on `scored` against `truth`.

    Run this on a validation set the matcher wasn't trained on (docs/normalize.md's rule applies here
    too: fit on sets you don't score on). Returns (best_threshold, best_f0.5).
    """
    # Coarse steps, then 0.01 steps near the top: tuned thresholds land at 0.9+ (docs/matching.md).
    thresholds = thresholds if thresholds is not None else ([i / 100 for i in range(5, 90, 5)] +
                                                            [i / 100 for i in range(90, 100)])
    best_t, best_f = thresholds[0], -1.0
    for t in thresholds:
        pred = {s1: set(g["cand_id"]) for s1, g in decide(scored, t, margin).groupby("s1_id")}
        f = evaluator.score(pred, truth)
        if f > best_f:
            best_t, best_f = t, f
    return best_t, best_f