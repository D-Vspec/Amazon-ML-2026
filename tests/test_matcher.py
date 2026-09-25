import numpy as np
import pandas as pd

from er.features import FEATURE_COLUMNS
from er.matcher import XgbMatcher

RNG = np.random.default_rng(0)


def labeled(columns, n=300):
    frame = pd.DataFrame(RNG.random((n, len(columns))).astype("float32"), columns=columns)
    return frame.assign(s1_id=[f"S1-{i}" for i in range(n)], cand_id=[f"S2-{i}" for i in range(n)],
                        label=(frame[columns[0]] > 0.5).astype(int))


def test_trains_on_every_feature_column():
    cols = FEATURE_COLUMNS + ["score_tfidf", "score_embedding", "found_by_all"]
    m = XgbMatcher(n_estimators=5).fit(labeled(cols))
    assert m.feature_names == cols
    assert list(m.feature_importance().sort_index().index) == sorted(cols)


def test_old_model_ignores_extra_columns(tmp_path):
    # A model trained on the 14 base features (like the teammate's model.json) still predicts on union features.
    XgbMatcher(n_estimators=5).fit(labeled(FEATURE_COLUMNS)).save(str(tmp_path / "m.json"))
    old = XgbMatcher.load(str(tmp_path / "m.json"))
    pairs = labeled(FEATURE_COLUMNS + ["score_tfidf", "score_embedding", "found_by_all"]).drop(columns="label")
    out = old.predict(pairs)
    assert old.feature_names == FEATURE_COLUMNS
    assert out["match_proba"].between(0, 1).all() and len(out) == len(pairs)


def test_save_load_round_trip(tmp_path):
    cols = FEATURE_COLUMNS + ["score_embedding"]
    data = labeled(cols)
    m = XgbMatcher(n_estimators=5).fit(data)
    m.save(str(tmp_path / "m.json"))
    loaded = XgbMatcher.load(str(tmp_path / "m.json"))
    assert loaded.feature_names == cols
    assert np.allclose(m.predict(data)["match_proba"], loaded.predict(data)["match_proba"])
