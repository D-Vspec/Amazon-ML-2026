import numpy as np
import pandas as pd
import pytest

import er.cross_encoder as ce_module
from er.cross_encoder import CrossEncoderScorer

SMALL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Apache-2.0, 118M: small enough for CPU tests
S1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "raw_name": ["Perfect Media", "Harris Better"],
                   "raw_address": ["C 1 A Sector 27, Noida", "8008 14 Ave, Brooklyn, NY"]})
TARGETS = pd.DataFrame({"entity_id": ["S2-1", "S2-2", "S3-1"],
                        "raw_name": ["परफेक्ट मीडिया", "Harris Better", "Global Infotech"],
                        "raw_address": ["C 1 A Sector 27, Noida", "8008 14 AVE, BROOKLYN", "Delhi"]})
PAIRS = pd.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-2", "S1-2"], "cand_id": ["S2-1", "S3-1", "S2-2", "S3-1"],
                      "score": [0.9, 0.3, 0.95, 0.1], "label": [1, 0, 1, 0]})


@pytest.fixture(scope="module")
def scorer():
    return CrossEncoderScorer("cpu", SMALL, batch_size=4)


def test_texts_join_raw_name_and_address_and_tolerate_missing_ids():
    pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-9"], "cand_id": ["S2-1", "S2-9"]})
    left, right = CrossEncoderScorer._texts(pairs, S1, TARGETS)
    assert left == ["Perfect Media | C 1 A Sector 27, Noida", ""]
    assert right == ["परफेक्ट मीडिया | C 1 A Sector 27, Noida", ""]


def test_score_is_a_probability_per_pair_in_order(scorer):
    scores = scorer.score(PAIRS, S1, TARGETS)
    assert scores.shape == (4,) and ((scores >= 0) & (scores <= 1)).all()
    reordered = scorer.score(PAIRS.iloc[::-1], S1, TARGETS)
    assert np.allclose(scores[::-1], reordered, atol=1e-5)


def test_fit_then_save_load_round_trip(tmp_path):
    scorer = CrossEncoderScorer("cpu", SMALL, batch_size=2, train_pairs=4).fit(PAIRS, S1, TARGETS)
    scorer.save(str(tmp_path / "ce"))
    loaded = CrossEncoderScorer.load(str(tmp_path / "ce"), "cpu", SMALL, batch_size=2)
    assert np.allclose(scorer.score(PAIRS, S1, TARGETS), loaded.score(PAIRS, S1, TARGETS), atol=1e-5)


def test_remote_code_only_for_reviewed_models_and_pinned(monkeypatch):
    calls = []
    monkeypatch.setattr(ce_module, "CrossEncoder", lambda path, **kw: calls.append((path, kw)))
    CrossEncoderScorer("cpu", "Alibaba-NLP/gte-multilingual-reranker-base")
    CrossEncoderScorer("cpu", SMALL)
    (_, gte_kwargs), (_, small_kwargs) = calls
    pin = ce_module.REVIEWED_REMOTE_CODE["Alibaba-NLP/gte-multilingual-reranker-base"]
    assert gte_kwargs["trust_remote_code"] and gte_kwargs["config_kwargs"]["code_revision"] == pin
    assert not small_kwargs["trust_remote_code"] and small_kwargs["config_kwargs"] == {}
