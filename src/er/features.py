"""Pair-level features for the matcher: similarity between an S1 record and one candidate.

Code: src/er/features.py. See docs/blocking.md (candidate contract: s1_id, cand_id, score) and
docs/normalize.md (input columns: name_norm, legal_form, address_norm).

Contract, matching the rest of the pipeline (RuleNormalizer, TfidfNgramBlocker):
    fit(...) -> self          # stateless, kept only so this slots into main.py the same way
    transform(pairs, s1_records, target_records) -> pairs + FEATURE_COLUMNS

s1_records / target_records must already carry entity_id, name_norm, legal_form, address_norm.
"""

import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "score", "rank", "score_gap",
    "name_jaccard", "name_token_set", "name_partial",
    "addr_jaccard", "addr_token_set",
    "legal_agree", "legal_missing",
    "num_jaccard", "num_overlap",
    "s1_addr_empty", "cand_addr_empty",
]


def blocker_columns(pairs: pd.DataFrame) -> list[str]:
    """Extra per-blocker columns a union blocker adds (score_<name>, found_by_all); passed through as features."""
    return [c for c in pairs.columns if c.startswith(("score_", "found_by_"))]


def _tok(s: str) -> set[str]:
    return set(s.split()) if s else set()


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _nums(s: str) -> set[str]:
    # address_norm keeps "/" and "-" only inside numbers (docs/normalize.md), so splitting on
    # them recovers house/PIN-style tokens without pulling in street-name words.
    return {t for t in s.replace("/", " ").replace("-", " ").split() if t.isdigit()}


class PairFeaturizer:
    """Turns blocker candidates into the feature rows the matcher trains and predicts on."""

    def fit(self, *_args) -> "PairFeaturizer":
        return self

    chunk_s1 = 20_000  # S1 records featurized at once; bounds memory (7.6M pairs in one go ran out of 22 GB)

    def transform(self, pairs: pd.DataFrame, s1_records: pd.DataFrame,
                 target_records: pd.DataFrame) -> pd.DataFrame:
        """pairs: s1_id, cand_id, score. Rows come back sorted best-first within each s1_id."""
        s1_cols = s1_records.set_index("entity_id")[["name_norm", "legal_form", "address_norm"]]
        t_cols = target_records.set_index("entity_id")[["name_norm", "legal_form", "address_norm"]]
        # Chunk by S1 so each S1's candidates stay together (rank and score_gap are per S1).
        s1_ids = pd.Series(pairs["s1_id"].unique()).sort_values().to_numpy()
        chunks = [self._transform_chunk(pairs[pairs["s1_id"].isin(s1_ids[i:i + self.chunk_s1])], s1_cols, t_cols)
                  for i in range(0, len(s1_ids), self.chunk_s1)]
        if not chunks:
            return pd.DataFrame(columns=["s1_id", "cand_id"] + FEATURE_COLUMNS + blocker_columns(pairs))
        return pd.concat(chunks, ignore_index=True)

    def _transform_chunk(self, pairs: pd.DataFrame, s1_cols: pd.DataFrame, t_cols: pd.DataFrame) -> pd.DataFrame:
        df = pairs.merge(s1_cols.add_prefix("s1_"), left_on="s1_id", right_index=True)
        df = df.merge(t_cols.add_prefix("cand_"), left_on="cand_id", right_index=True)
        df = df.sort_values(["s1_id", "score"], ascending=[True, False]).reset_index(drop=True)

        # Rank and gap-from-best are cheap and useful: a right answer is usually the top or
        # near-top candidate for its S1, so a low rank / small gap is itself evidence.
        df["rank"] = df.groupby("s1_id").cumcount()
        df["score_gap"] = df["score"] - df.groupby("s1_id")["score"].transform("first")

        s1_name_tok = df["s1_name_norm"].map(_tok)
        cand_name_tok = df["cand_name_norm"].map(_tok)
        s1_addr_tok = df["s1_address_norm"].map(_tok)
        cand_addr_tok = df["cand_address_norm"].map(_tok)

        df["name_jaccard"] = [_jaccard(a, b) for a, b in zip(s1_name_tok, cand_name_tok)]
        df["name_token_set"] = [fuzz.token_set_ratio(a, b) / 100
                                for a, b in zip(df.s1_name_norm, df.cand_name_norm)]
        # partial_ratio catches a match hiding inside a longer, noisier name
        # ("services esquivel vonk illumination p.c." containing "esquivel vonk illumination").
        df["name_partial"] = [fuzz.partial_ratio(a, b) / 100
                              for a, b in zip(df.s1_name_norm, df.cand_name_norm)]

        df["addr_jaccard"] = [_jaccard(a, b) for a, b in zip(s1_addr_tok, cand_addr_tok)]
        df["addr_token_set"] = [fuzz.token_set_ratio(a, b) / 100
                                for a, b in zip(df.s1_address_norm, df.cand_address_norm)]

        # legal_form is a space-joined, sorted, deduped string (e.g. "ltd pvt"); compare as sets
        # so token order never matters.
        s1_legal = df["s1_legal_form"].map(_tok)
        cand_legal = df["cand_legal_form"].map(_tok)
        df["legal_agree"] = [float(bool(a) and a == b) for a, b in zip(s1_legal, cand_legal)]
        df["legal_missing"] = [float(not a or not b) for a, b in zip(s1_legal, cand_legal)]

        s1_nums = df["s1_address_norm"].map(_nums)
        cand_nums = df["cand_address_norm"].map(_nums)
        df["num_jaccard"] = [_jaccard(a, b) for a, b in zip(s1_nums, cand_nums)]
        df["num_overlap"] = [float(bool(a & b)) for a, b in zip(s1_nums, cand_nums)]

        df["s1_addr_empty"] = (df["s1_address_norm"] == "").astype(float)
        df["cand_addr_empty"] = (df["cand_address_norm"] == "").astype(float)

        return df[["s1_id", "cand_id"]].join(df[FEATURE_COLUMNS + blocker_columns(pairs)].astype("float32"))