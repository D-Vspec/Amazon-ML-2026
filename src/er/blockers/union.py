"""Union of blockers: each proposes its own top-k, and the lists are merged. See docs/blocking.md.

Recall of the union is "found by ANY member". Every member scores every merged pair (score_pairs), so each
pair carries one real similarity per member plus found_by_all, and the matcher can learn how far to trust
each blocker.
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
        """s1_id, cand_id, score (best member similarity), score_<name> and found_by_<name> per member, and
        found_by_all; best first per S1."""
        merged = None
        for name, blocker in self.blockers.items():
            pairs = blocker.query(s1_records, k).rename(columns={"score": f"score_{name}"})
            merged = pairs if merged is None else merged.merge(pairs, on=["s1_id", "cand_id"], how="outer")
        member_scores = [f"score_{name}" for name in self.blockers]
        member_found = [f"found_by_{name}" for name in self.blockers]
        for name in self.blockers:  # which member put the pair in its own top k
            merged[f"found_by_{name}"] = merged[f"score_{name}"].notna().astype("float32")
        merged["found_by_all"] = merged[member_found].min(axis=1)
        # Every member scores every pair, so a missing score means "not in my top k", never "similarity 0".
        for name, blocker in self.blockers.items():
            merged[f"score_{name}"] = blocker.score_pairs(merged) if len(merged) else merged[f"score_{name}"]
        merged["score"] = merged[member_scores].max(axis=1)
        merged = merged.sort_values(["s1_id", "score"], ascending=[True, False], kind="stable")
        columns = ["s1_id", "cand_id", "score", *member_scores, *member_found, "found_by_all"]
        return merged[columns].reset_index(drop=True)
