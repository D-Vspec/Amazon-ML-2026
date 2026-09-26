import pandas as pd

from er.cache import PairCache

PAIRS = pd.DataFrame({"s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-1", "S3-4"], "score": [0.9, 0.4]}).astype(
    {"score": "float32"})
S1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "raw_name": ["राम", "Acme"], "raw_address": ["Delhi", ""],
                   "name_norm": ["ram", "acme"]})
TARGETS = pd.DataFrame({"entity_id": ["S2-1", "S3-4"], "raw_name": ["Ram", "X"], "raw_address": ["DL", "Y"],
                        "country": ["India", "US"]})
SETTINGS = {"BLOCKER": "tfidf+embedding", "BLOCK_K": "20"}


def test_miss_then_hit_round_trip(tmp_path):
    cache = PairCache(tmp_path, SETTINGS)
    assert cache.load([3]) is None
    cache.save([3], PAIRS, S1, TARGETS)
    pairs, s1, targets = cache.load([3])
    pd.testing.assert_frame_equal(pairs, PAIRS)
    assert list(s1.columns) == ["entity_id", "raw_name", "raw_address"]  # only what the cross-encoder needs
    assert s1.raw_name.tolist() == ["राम", "Acme"] and targets.entity_id.tolist() == ["S2-1", "S3-4"]


def test_different_settings_or_sets_miss(tmp_path):
    PairCache(tmp_path, SETTINGS).save([3], PAIRS, S1, TARGETS)
    assert PairCache(tmp_path, {**SETTINGS, "BLOCK_K": "10"}).load([3]) is None
    assert PairCache(tmp_path, SETTINGS).load([2]) is None
    assert PairCache(tmp_path, SETTINGS).load([3, 4]) is None


def test_extra_column_round_trip(tmp_path):
    cache = PairCache(tmp_path, SETTINGS)
    cache.save([3], PAIRS, S1, TARGETS)
    assert cache.load_column([3], "score_ce_a") is None
    cache.save_column([3], "score_ce_a", [0.25, 0.75])
    assert cache.load_column([3], "score_ce_a").tolist() == [0.25, 0.75]
    assert cache.load_column([3], "score_ce_b") is None  # another model's scores are a different column


def test_interrupted_write_is_not_read(tmp_path):
    cache = PairCache(tmp_path, SETTINGS)
    cache.save([3], PAIRS, S1, TARGETS)
    (cache.dir / "sets_3" / "done").unlink()  # as if the process died before finishing
    assert cache.load([3]) is None
