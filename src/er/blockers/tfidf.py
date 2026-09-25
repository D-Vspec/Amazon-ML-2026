"""TF-IDF over character trigrams of name + address; nearest neighbours by cosine. See docs/blocking.md."""

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

# The CSR matrix comes from scipy (always valid), and sparse CSR being "beta" in torch is informational.
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

PAIR_COLUMNS = ["s1_id", "cand_id", "score"]
MAX_SCORES = 250_000_000  # float32 scores held at once per batch (~1 GB)
PAIR_CHUNK = 1_000_000  # pairs rescored at once in score_pairs


class TfidfNgramBlocker:
    def __init__(self, device: str = "cpu", ngram_range=(3, 3), max_df: float = 0.05):
        # max_df drops trigrams found in >5% of records: 2x faster, same recall (docs/blocking.md).
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, max_df=max_df,
                                          sublinear_tf=True, dtype=np.float32)

    @staticmethod
    def _text(records: pd.DataFrame) -> pd.Series:
        return records["name_norm"] + " " + records["address_norm"]

    def fit(self, targets: pd.DataFrame) -> "TfidfNgramBlocker":
        """Index the S2+S3 records, one sparse matrix per country."""
        matrix = self.vectorizer.fit_transform(self._text(targets))
        ids = targets["entity_id"].to_numpy()
        self._target_matrix, self._target_row = matrix, pd.Index(ids)  # kept for score_pairs
        self.targets = {}
        for country, rows in targets.groupby("country").indices.items():
            m = matrix[rows].tocsr()
            tensor = torch.sparse_csr_tensor(torch.from_numpy(m.indptr).long(), torch.from_numpy(m.indices).long(),
                                             torch.from_numpy(m.data), size=m.shape, device=self.device)
            self.targets[country] = (tensor, ids[rows])
        return self

    def query(self, s1_records: pd.DataFrame, k: int) -> pd.DataFrame:
        """Top-k same-country targets per S1 by cosine similarity, best first. Zero-similarity pairs are dropped."""
        queries = self.vectorizer.transform(self._text(s1_records))
        s1_ids = s1_records["entity_id"].to_numpy()
        self._query_matrix, self._query_row = queries, pd.Index(s1_ids)  # kept for score_pairs
        frames = []
        for country, rows in s1_records.groupby("country").indices.items():
            if country not in self.targets:  # no S2/S3 records from this country: no candidates
                continue
            targets, target_ids = self.targets[country]
            top_k = min(k, len(target_ids))
            batch = max(1, MAX_SCORES // len(target_ids))
            for start in range(0, len(rows), batch):
                chunk = rows[start:start + batch]
                q = torch.from_numpy(queries[chunk].T.toarray()).to(self.device)  # (vocab, batch)
                best = torch.topk(torch.sparse.mm(targets, q), top_k, dim=0)  # scores are (targets, batch)
                frames.append(pd.DataFrame({
                    "s1_id": np.repeat(s1_ids[chunk], top_k),
                    "cand_id": target_ids[best.indices.T.cpu().numpy().ravel()],
                    "score": best.values.T.cpu().numpy().ravel(),
                }))
        if not frames:
            return pd.DataFrame(columns=PAIR_COLUMNS)
        pairs = pd.concat(frames, ignore_index=True)
        return pairs[pairs["score"] > 0].reset_index(drop=True)

    def score_pairs(self, pairs: pd.DataFrame) -> np.ndarray:
        """Cosine similarity for any (s1_id, cand_id) pairs from the last fit/query, not just this blocker's top k."""
        qi = self._query_row.get_indexer(pairs["s1_id"])
        ti = self._target_row.get_indexer(pairs["cand_id"])
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), PAIR_CHUNK):
            end = start + PAIR_CHUNK
            rows = self._query_matrix[qi[start:end]].multiply(self._target_matrix[ti[start:end]])
            out[start:end] = np.asarray(rows.sum(axis=1)).ravel()
        return out
