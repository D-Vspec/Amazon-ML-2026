"""Write matching_results.tsv and candidate_pairs.tsv in the exact submission format.

Code: src/er/output.py. See the problem statement's Output Format section and utils/validate_submission.py
(run that after this, before ever uploading).

Every s1_id in `s1_ids` gets exactly one row, in that order, with an empty string when it has no ids
(a singleton, or an S1 whose country has no candidates at all — docs/blocking.md).
"""

from typing import Iterable

import pandas as pd


def _write_id_lists(df: pd.DataFrame, out_col: str, s1_ids: Iterable[str], path: str) -> None:
    """df must have columns s1_id, cand_id. Deduplicates ids within each s1 before joining."""
    grouped = (df.drop_duplicates(["s1_id", "cand_id"])
                 .groupby("s1_id")["cand_id"]
                 .apply(lambda s: ",".join(s)))
    grouped = grouped.reindex(list(s1_ids), fill_value="")
    grouped.rename(out_col).rename_axis("source1_entity_id").reset_index().to_csv(
        path, sep="\t", index=False)


def write_matching_results(matches: pd.DataFrame, s1_ids: Iterable[str], path: str) -> None:
    """matches: s1_id, cand_id (decision.decide's output, or any two columns matching that shape)."""
    _write_id_lists(matches, "matched_entity_ids", s1_ids, path)


def write_candidate_pairs(candidates: pd.DataFrame, s1_ids: Iterable[str], path: str) -> None:
    """candidates: s1_id, cand_id (the blocker's raw output — the exact list fed to the matcher)."""
    _write_id_lists(candidates, "candidate_entity_ids", s1_ids, path)   