"""Union of blockers: each proposes its own top-k, and the lists are merged. See docs/blocking.md.

Recall of the union is "found by ANY member". Each pair keeps one score column per member (0 when that
member didn't propose it), so the matcher can learn how much to trust each blocker.
"""

import pandas as pd


class UnionBlocker:
    def __init__(self, blockers: dict):
        self.blockers = blockers  # name -> blocker instance
        self.device = next(iter(blockers.values())).device

    def fit(self, targets: pd.DataFrame) -> "UnionBlocker":
        for blocker in self.blockers.values():
            blocker.fit(targets)
        return self

    def query(self, s1_records: pd.DataFrame, k: int) -> pd.DataFrame:
        """s1_id, cand_id, score (best of the members), score_<name> per member; best first per S1."""
        merged = None
        for name, blocker in self.blockers.items():
            pairs = blocker.query(s1_records, k).rename(columns={"score": f"score_{name}"})
            merged = pairs if merged is None else merged.merge(pairs, on=["s1_id", "cand_id"], how="outer")
        member_scores = [f"score_{name}" for name in self.blockers]
        merged[member_scores] = merged[member_scores].fillna(0.0)
        merged["score"] = merged[member_scores].max(axis=1)
        merged = merged.sort_values(["s1_id", "score"], ascending=[True, False], kind="stable")
        return merged[["s1_id", "cand_id", "score", *member_scores]].reset_index(drop=True)
