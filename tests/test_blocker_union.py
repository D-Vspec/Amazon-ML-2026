import pandas as pd

from er.blockers.union import UnionBlocker


class FakeBlocker:
    """Returns fixed pairs; records whether fit was called."""
    device = "cpu"

    def __init__(self, pairs):
        self.pairs = pd.DataFrame(pairs, columns=["s1_id", "cand_id", "score"])
        self.fitted = False

    def fit(self, targets):
        self.fitted = True
        return self

    def query(self, s1_records, k):
        return self.pairs.groupby("s1_id").head(k).reset_index(drop=True)


def union(a_pairs, b_pairs):
    return UnionBlocker({"a": FakeBlocker(a_pairs), "b": FakeBlocker(b_pairs)})


def test_fit_fits_every_member():
    u = union([], [])
    u.fit(pd.DataFrame())
    assert all(b.fitted for b in u.blockers.values())


def test_union_dedupes_and_keeps_both_scores():
    u = union([("S1-1", "S2-1", 0.9), ("S1-1", "S2-2", 0.5)],
              [("S1-1", "S2-1", 0.7), ("S1-1", "S3-9", 0.8)])
    out = u.query(pd.DataFrame(), 5)
    assert list(out.columns) == ["s1_id", "cand_id", "score", "score_a", "score_b"]
    rows = {r.cand_id: (r.score, r.score_a, r.score_b) for r in out.itertuples()}
    assert rows == {"S2-1": (0.9, 0.9, 0.7), "S3-9": (0.8, 0.0, 0.8), "S2-2": (0.5, 0.5, 0.0)}


def test_sorted_best_first_per_s1():
    u = union([("S1-2", "S2-1", 0.2), ("S1-1", "S2-5", 0.4)], [("S1-1", "S2-6", 0.95), ("S1-2", "S2-7", 0.6)])
    out = u.query(pd.DataFrame(), 5)
    assert list(out.cand_id) == ["S2-6", "S2-5", "S2-7", "S2-1"]


def test_each_member_keeps_its_own_k():
    u = union([("S1-1", f"S2-{i}", 1 - i / 10) for i in range(5)], [("S1-1", f"S3-{i}", 1 - i / 10) for i in range(5)])
    out = u.query(pd.DataFrame(), 2)
    assert set(out.cand_id) == {"S2-0", "S2-1", "S3-0", "S3-1"}  # up to 2k per S1


def test_empty_members():
    out = union([], []).query(pd.DataFrame(), 3)
    assert out.empty and list(out.columns) == ["s1_id", "cand_id", "score", "score_a", "score_b"]
