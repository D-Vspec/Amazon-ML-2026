"""Pair-level features for the matcher: similarity between an S1 record and one candidate.

Code: src/er/features.py. See docs/blocking.md (candidate contract: s1_id, cand_id, score) and
docs/normalize.md (input columns: name_norm, legal_form, address_norm).

Contract, matching the rest of the pipeline (RuleNormalizer, TfidfNgramBlocker):
    fit(...) -> self          # stateless, kept only so this slots into main.py the same way
    transform(pairs, s1_records, target_records) -> pairs + FEATURE_COLUMNS

s1_records / target_records must already carry entity_id, name_norm, legal_form, address_norm.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

FEATURE_COLUMNS = [
    "score", "rank", "score_gap",
    "name_jaccard", "name_token_set", "name_partial",
    "name_extra_tokens",
    "addr_jaccard", "addr_token_set",
    "legal_agree", "legal_missing",
    "num_jaccard", "num_overlap",
    "s1_addr_empty", "cand_addr_empty",
]


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

    def transform(self, pairs: pd.DataFrame, s1_records: pd.DataFrame,
                 target_records: pd.DataFrame) -> pd.DataFrame:
        """pairs: s1_id, cand_id, score. Rows come back sorted best-first within each s1_id."""
        s1_cols = s1_records.set_index("entity_id")[["name_norm", "legal_form", "address_norm"]]
        t_cols = target_records.set_index("entity_id")[["name_norm", "legal_form", "address_norm"]]

        df = pairs.merge(s1_cols.add_prefix("s1_"), left_on="s1_id", right_index=True)
        df = df.merge(t_cols.add_prefix("cand_"), left_on="cand_id", right_index=True)
        df = df.sort_values(["s1_id", "score"], ascending=[True, False]).reset_index(drop=True)

        # Rank and gap-from-best are cheap and useful: a right answer is usually the top or
        # near-top candidate for its S1, so a low rank / small gap is itself evidence.
        df["rank"] = df.groupby("s1_id").cumcount()
        df["score_gap"] = df["score"] - df.groupby("s1_id")["score"].transform("first")

        # Every s1_id repeats ~BLOCK_K times and every cand_id repeats across all the S1s that
        # matched it, so tokenizing/parsing once per unique id -- instead of once per pair -- cuts
        # the Python-level work by roughly that repeat factor. Build one small lookup table per
        # side, on the unique (name_norm, address_norm, legal_form) rows, then map back onto df.
        def side_lookup(id_col: str, name_col: str, addr_col: str, legal_col: str) -> pd.DataFrame:
            uniq = df[[id_col, name_col, addr_col, legal_col]].drop_duplicates(id_col).set_index(id_col)
            out = pd.DataFrame(index=uniq.index)
            out["name_tok"] = uniq[name_col].map(_tok)
            out["addr_tok"] = uniq[addr_col].map(_tok)
            out["nums"] = uniq[addr_col].map(_nums)
            out["legal_tok"] = uniq[legal_col].map(_tok)
            return out

        s1_lookup = side_lookup("s1_id", "s1_name_norm", "s1_address_norm", "s1_legal_form")
        cand_lookup = side_lookup("cand_id", "cand_name_norm", "cand_address_norm", "cand_legal_form")

        s1_name_tok = df["s1_id"].map(s1_lookup["name_tok"])
        cand_name_tok = df["cand_id"].map(cand_lookup["name_tok"])
        s1_addr_tok = df["s1_id"].map(s1_lookup["addr_tok"])
        cand_addr_tok = df["cand_id"].map(cand_lookup["addr_tok"])
        s1_nums = df["s1_id"].map(s1_lookup["nums"])
        cand_nums = df["cand_id"].map(cand_lookup["nums"])
        s1_legal = df["s1_id"].map(s1_lookup["legal_tok"])
        cand_legal = df["cand_id"].map(cand_lookup["legal_tok"])

        df["name_jaccard"] = [_jaccard(a, b) for a, b in zip(s1_name_tok, cand_name_tok)]
        df["addr_jaccard"] = [_jaccard(a, b) for a, b in zip(s1_addr_tok, cand_addr_tok)]
        df["num_jaccard"] = [_jaccard(a, b) for a, b in zip(s1_nums, cand_nums)]
        df["num_overlap"] = [float(bool(a & b)) for a, b in zip(s1_nums, cand_nums)]

        # Raw count of tokens present in ONE name but not the other (legal_form is a separate field,
        # so these are real content words -- "Group", "Industries", "Exports", a mis-transliterated
        # word, etc). This targets a specific adversarial pattern seen in error analysis: near-
        # identical name + address, differing by exactly one such word, where the true label is
        # "not a match". name_jaccard dilutes a single differing word when the rest of a long name
        # matches (e.g. "pai seva samiti" vs "pai seva samiti exports" still scores jaccard=0.75);
        # a raw, un-normalized extra-token count gives the tree a sharper, non-diluted signal to
        # split on instead of relying on jaccard/token_set/partial to carry it indirectly.
        df["name_extra_tokens"] = [float(len(a ^ b)) for a, b in zip(s1_name_tok, cand_name_tok)]

        # legal_form is a space-joined, sorted, deduped string (e.g. "ltd pvt"); compare as sets
        # so token order never matters.
        df["legal_agree"] = [float(bool(a) and a == b) for a, b in zip(s1_legal, cand_legal)]
        df["legal_missing"] = [float(not a or not b) for a, b in zip(s1_legal, cand_legal)]

        # Fuzzy scores (token_set_ratio / partial_ratio): scoring one pair at a time means
        # 3 rapidfuzz calls x len(df) individual Python calls (~13M at full data size), which is
        # dominated by call overhead rather than the underlying C work. Each S1 only has BLOCK_K
        # candidates, so group by s1_id and score a whole group against its one S1 string in a
        # single process.cdist call instead -- same computation, far fewer Python-level calls.
        name_token_set = np.empty(len(df))
        name_partial = np.empty(len(df))
        addr_token_set = np.empty(len(df))

        s1_name_arr = df["s1_name_norm"].to_numpy()
        cand_name_arr = df["cand_name_norm"].to_numpy()
        s1_addr_arr = df["s1_address_norm"].to_numpy()
        cand_addr_arr = df["cand_address_norm"].to_numpy()

        for idx in df.groupby("s1_id", sort=False).indices.values():
            idx = np.asarray(idx)
            s1_name = s1_name_arr[idx[0]]
            s1_addr = s1_addr_arr[idx[0]]
            name_token_set[idx] = process.cdist([s1_name], cand_name_arr[idx], scorer=fuzz.token_set_ratio)[0]
            name_partial[idx] = process.cdist([s1_name], cand_name_arr[idx], scorer=fuzz.partial_ratio)[0]
            addr_token_set[idx] = process.cdist([s1_addr], cand_addr_arr[idx], scorer=fuzz.token_set_ratio)[0]

        df["name_token_set"] = name_token_set / 100
        df["name_partial"] = name_partial / 100
        df["addr_token_set"] = addr_token_set / 100

        df["s1_addr_empty"] = (df["s1_address_norm"] == "").astype(float)
        df["cand_addr_empty"] = (df["cand_address_norm"] == "").astype(float)

        return df[["s1_id", "cand_id"] + FEATURE_COLUMNS]