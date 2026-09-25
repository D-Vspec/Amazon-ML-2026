"""Gradient-boosted classifier that scores each candidate pair's probability of being a true match.

Code: src/er/matcher.py. See docs/matching.md. Trains on labeled pairs from training_pairs.py, using
the FEATURE_COLUMNS from features.py.

Contract, matching the rest of the pipeline: fit(labeled_pairs) -> self, predict(pairs) -> pairs +
`match_proba`. MIT-licensed (xgboost), well under the 8B-parameter limit.
"""

import pandas as pd
import xgboost as xgb

from .features import FEATURE_COLUMNS


class XgbMatcher:
    def __init__(self, **params):
        # scale_pos_weight is set from the training data at fit() time when left as None: the true
        # class balance depends on BLOCK_K and isn't known until then (docs/blocking.md: k=20 candidates
        # per S1, ~3.46 true matches per S1, so roughly 1 positive per 5 negatives at k=20, but that
        # shifts with k).
        defaults = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                        scale_pos_weight=None, n_jobs=-1, random_state=0)
        self.params = {**defaults, **params}
        self.model: xgb.XGBClassifier | None = None

    def fit(self, labeled_pairs: pd.DataFrame) -> "XgbMatcher":
        X, y = labeled_pairs[FEATURE_COLUMNS], labeled_pairs["label"]
        params = dict(self.params)
        if params["scale_pos_weight"] is None:
            neg, pos = int((y == 0).sum()), int((y == 1).sum())
            params["scale_pos_weight"] = neg / pos if pos else 1.0
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        return self

    def predict(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Adds `match_proba` (probability the pair is a true match) to a copy of `pairs`."""
        out = pairs.copy()
        out["match_proba"] = self.model.predict_proba(pairs[FEATURE_COLUMNS])[:, 1]
        return out

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)