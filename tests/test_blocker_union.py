import pandas as pd

from er.blockers.union import UnionBlocker


class FakeBlocker:
    """Proposes fixed pairs; score_pairs looks similarities up in a table (proposed pairs use their own score)."""
    device = "cpu"

    def __init__(self, pairs, similarity=None):
        self.pairs = pd.DataFrame(pairs, columns=["s1_id", "cand_id", "score"])
        self.similarity = {(s, c): v for s, c, v in pairs} | (similarity or {})
        self.fitted = False

    def fit(self, targets):
        self.fitted = True
        return self

    def query(self, s1_records, k):
        return self.pairs.groupby("s1_id").head(k).reset_index(drop=True)

    def score_pairs(self, pairs):
        return [self.similarity.get((s, c), 0.0) for s, c in zip(pairs.s1_id, pairs.cand_id)]


COLUMNS = ["s1_id", "cand_id", "score", "score_a", "score_b", "found_by_all"]


def union(a_pairs, b_pairs, a_sim=None, b_sim=None):
    return UnionBlocker({"a": FakeBlocker(a_pairs, a_sim), "b": FakeBlocker(b_pairs, b_sim)})


def test_fit_fits_every_member():
    u = union([], [])
    u.fit(pd.DataFrame())
    assert all(b.fitted for b in u.blockers.values())


def test_every_member_scores_every_pair():
    u = union([("S1-1", "S2-1", 0.9), ("S1-1", "S2-2", 0.5)],
              [("S1-1", "S2-1", 0.7), ("S1-1", "S3-9", 0.8)],
              a_sim={("S1-1", "S3-9"): 0.2}, b_sim={("S1-1", "S2-2"): 0.3})
    out = u.query(pd.DataFrame(), 5)
    assert list(out.columns) == COLUMNS
    rows = {r.cand_id: (r.score, r.score_a, r.score_b, r.found_by_all) for r in out.itertuples()}
    # S3-9 was only proposed by b, yet gets a's real similarity (0.2), not a 0 placeholder.
    assert rows == {"S2-1": (0.9, 0.9, 0.7, 1.0), "S3-9": (0.8, 0.2, 0.8, 0.0), "S2-2": (0.5, 0.5, 0.3, 0.0)}


def test_score_is_best_member_similarity():
    u = union([("S1-1", "S2-1", 0.4)], [("S1-1", "S3-1", 0.6)], a_sim={("S1-1", "S3-1"): 0.95})
    out = u.query(pd.DataFrame(), 5).set_index("cand_id")
    assert out.loc["S3-1", "score"] == 0.95  # a's real similarity beats b's proposal score


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
    assert out.empty and list(out.columns) == COLUMNS
