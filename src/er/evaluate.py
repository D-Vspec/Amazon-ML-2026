"""Local scoring: the leaderboard's macro F0.5, plus blocking recall. See docs/evaluation.md."""

from pathlib import Path

IdLists = dict[str, set[str]]


class F05Evaluator:
    beta2 = 0.25  # beta = 0.5

    @staticmethod
    def load(path: Path) -> IdLists:
        """Read a `source1_entity_id <TAB> comma-separated ids` file (ground truth, matches or candidates)."""
        out = {}
        with open(path, encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                s1, _, ids = line.rstrip("\n").partition("\t")
                out[s1] = {x for x in ids.split(",") if x}
        return out

    def entity_score(self, pred: set[str], truth: set[str]) -> float:
        if not truth:  # singleton: full credit only for predicting nothing
            return 1.0 if not pred else 0.0
        hits = len(pred & truth)
        if hits == 0:
            return 0.0
        p, r = hits / len(pred), hits / len(truth)
        return (1 + self.beta2) * p * r / (self.beta2 * p + r)

    def score(self, pred: IdLists, truth: IdLists) -> float:
        """Macro F0.5 over every S1 entity in `truth`. An S1 missing from `pred` counts as an empty prediction."""
        return sum(self.entity_score(pred.get(s1, set()), ids) for s1, ids in truth.items()) / len(truth)

    @staticmethod
    def blocking_recall(candidates: IdLists, truth: IdLists) -> float:
        """Share of all true (S1, match) pairs that appear in the candidates: the ceiling on matcher recall."""
        total = sum(len(ids) for ids in truth.values())
        found = sum(len(ids & candidates.get(s1, set())) for s1, ids in truth.items())
        return found / total if total else 1.0

    @staticmethod
    def mean_candidates(candidates: IdLists, truth: IdLists) -> float:
        """Average candidate-list length per S1 in `truth` (the matcher's workload)."""
        return sum(len(candidates.get(s1, ())) for s1 in truth) / len(truth)
