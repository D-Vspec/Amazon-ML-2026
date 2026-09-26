"""TF-IDF over character trigrams of name + address; nearest neighbours by cosine. See docs/blocking.md.

Patch note: fit() used to build every country's sparse tensor on the GPU up front and hold them all
resident for the life of the blocker. At full-test scale (millions of targets per country) that can
exceed an 8GB card once all countries are summed together, and CUDA silently pages to system RAM
instead of erroring — symptoms: low GPU utilization, near-full GPU memory, no progress. fit() now
keeps each country's matrix on CPU (scipy CSR); query() builds the GPU tensor for one country at a
time and frees it before moving to the next, so peak VRAM is bounded by the single largest country,
not the sum of all of them.
"""

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

PAIR_COLUMNS = ["s1_id", "cand_id", "score"]
MAX_SCORES = 250_000_000  # float32 scores held at once per batch (~1 GB)
MAX_QUERY_ROWS_PER_BATCH = 100_000  # hard cap on rows densified per chunk, regardless of target count.


class TfidfNgramBlocker:
    def __init__(self, device: str = "cpu", ngram_range=(3, 3), max_df: float = 0.05):
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, max_df=max_df,
                                          sublinear_tf=True, dtype=np.float32)

    @staticmethod
    def _text(records: pd.DataFrame) -> pd.Series:
        return records["name_norm"] + " " + records["address_norm"]

    def fit(self, targets: pd.DataFrame) -> "TfidfNgramBlocker":
        """Index the S2+S3 records. Kept on CPU as scipy CSR, one matrix per country — see the module
        docstring for why this changed from building GPU tensors here."""
        matrix = self.vectorizer.fit_transform(self._text(targets))
        ids = targets["entity_id"].to_numpy()
        self.targets_cpu = {}
        for country, rows in targets.groupby("country").indices.items():
            self.targets_cpu[country] = (matrix[rows].tocsr(), ids[rows])
        return self

    def _to_gpu(self, m):
        return torch.sparse_csr_tensor(torch.from_numpy(m.indptr).long(), torch.from_numpy(m.indices).long(),
                                       torch.from_numpy(m.data), size=m.shape, device=self.device)

    def query(self, s1_records: pd.DataFrame, k: int) -> pd.DataFrame:
        """Top-k same-country targets per S1 by cosine similarity, best first. Zero-similarity pairs are dropped."""
        queries = self.vectorizer.transform(self._text(s1_records))
        s1_ids = s1_records["entity_id"].to_numpy()
        frames = []
        for country, rows in s1_records.groupby("country").indices.items():
            if country not in self.targets_cpu:  # no S2/S3 records from this country: no candidates
                continue
            m, target_ids = self.targets_cpu[country]
            targets = self._to_gpu(m)  # only this country's matrix is on the GPU at any time
            top_k = min(k, len(target_ids))
            batch = max(1, min(MAX_SCORES // len(target_ids), MAX_QUERY_ROWS_PER_BATCH))
            country_s1, country_cand, country_score = [], [], []
            for start in range(0, len(rows), batch):
                chunk = rows[start:start + batch]
                q = torch.from_numpy(queries[chunk].T.toarray()).to(self.device)
                best = torch.topk(torch.sparse.mm(targets, q), top_k, dim=0)
                country_s1.append(np.repeat(s1_ids[chunk], top_k))
                country_cand.append(target_ids[best.indices.T.cpu().numpy().ravel()])
                country_score.append(best.values.T.cpu().numpy().ravel())
            del targets  # free this country's GPU memory before the next one is built
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if country_s1:
                frames.append(pd.DataFrame({
                    "s1_id": np.concatenate(country_s1),
                    "cand_id": np.concatenate(country_cand),
                    "score": np.concatenate(country_score),
                }))
        if not frames:
            return pd.DataFrame(columns=PAIR_COLUMNS)
        pairs = pd.concat(frames, ignore_index=True)
        return pairs[pairs["score"] > 0].reset_index(drop=True)