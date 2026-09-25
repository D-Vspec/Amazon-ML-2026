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
