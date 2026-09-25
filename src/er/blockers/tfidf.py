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
MAX_QUERY_ROWS_PER_BATCH = 100_000  # hard cap on rows densified per chunk, regardless of target count.
# Bounded, but not so small that per-batch Python overhead (a GPU sync + a DataFrame build every
# batch) dominates wall time when there are millions of query rows split across many countries.
TARGET_CHUNK_ROWS = 200_000
# Cap on target rows moved to GPU at once, *per country*. A single country (e.g. the largest by
# record count) can itself hold millions of rows, and its sparse tensor alone can exceed GPU memory
# even after query batching -- this bounds the target side the same way MAX_QUERY_ROWS_PER_BATCH
# bounds the query side, independent of how large any one country is.


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
        """Index the S2+S3 records, sharded into row-chunks per country.

        Tensors are built and kept on CPU, not self.device -- holding every country's (or even one
        large country's) full sparse tensor on GPU at once is what exhausts GPU memory with
        millions of target rows. self.targets[country] is a list of (cpu_tensor, ids_chunk) shards,
        each capped at TARGET_CHUNK_ROWS rows, so query() can move one bounded shard to GPU at a
        time instead of a whole country's index.
        """
        matrix = self.vectorizer.fit_transform(self._text(targets))
        ids = targets["entity_id"].to_numpy()
        self.targets = {}
        for country, rows in targets.groupby("country").indices.items():
            shards = []
            for start in range(0, len(rows), TARGET_CHUNK_ROWS):
                chunk_rows = rows[start:start + TARGET_CHUNK_ROWS]
                m = matrix[chunk_rows].tocsr()
                tensor = torch.sparse_csr_tensor(torch.from_numpy(m.indptr).long(),
                                                 torch.from_numpy(m.indices).long(),
                                                 torch.from_numpy(m.data), size=m.shape, device="cpu")
                shards.append((tensor, ids[chunk_rows]))
            self.targets[country] = shards
        return self

    def query(self, s1_records: pd.DataFrame, k: int) -> pd.DataFrame:
        """Top-k same-country targets per S1 by cosine similarity, best first. Zero-similarity pairs are dropped."""
        queries = self.vectorizer.transform(self._text(s1_records))
        s1_ids = s1_records["entity_id"].to_numpy()
        frames = []
        for country, rows in s1_records.groupby("country").indices.items():
            if country not in self.targets:  # no S2/S3 records from this country: no candidates
                continue
            shards = self.targets[country]
            total_targets = sum(len(ids) for _, ids in shards)
            top_k = min(k, total_targets)
            # Query batch size: same capping as before, based on total target count across all
            # shards of this country (keeps the per-shard score matrix ~1GB, MAX_SCORES) and by a
            # flat row cap (MAX_QUERY_ROWS_PER_BATCH) so a country with few targets but many S1
            # query rows never densifies a huge chunk in one shot.
            batch = max(1, min(MAX_SCORES // total_targets, MAX_QUERY_ROWS_PER_BATCH))
            country_s1, country_cand, country_score = [], [], []
            for start in range(0, len(rows), batch):
                chunk = rows[start:start + batch]
                q = torch.from_numpy(queries[chunk].T.toarray()).to(self.device)  # (vocab, batch)
                running_vals, running_ids = None, None  # (t, len(chunk)) running top-k across shards
                for cpu_t, ids_shard in shards:
                    t = cpu_t.to(self.device)
                    scores = torch.sparse.mm(t, q)  # (shard_rows, len(chunk))
                    del t
                    cur_k = min(top_k, scores.shape[0])
                    vals, idxs = torch.topk(scores, cur_k, dim=0)
                    del scores
                    ids_t = torch.from_numpy(ids_shard).to(self.device)
                    cand_ids = ids_t[idxs]  # (cur_k, len(chunk)) entity ids for this shard's top hits
                    del idxs, ids_t
                    if running_vals is None:
                        running_vals, running_ids = vals, cand_ids
                    else:
                        merged_vals = torch.cat([running_vals, vals], dim=0)
                        merged_ids = torch.cat([running_ids, cand_ids], dim=0)
                        del running_vals, running_ids, vals, cand_ids
                        merge_k = min(top_k, merged_vals.shape[0])
                        merged_top_vals, sel = torch.topk(merged_vals, merge_k, dim=0)
                        running_vals = merged_top_vals
                        running_ids = torch.gather(merged_ids, 0, sel)
                        del merged_vals, merged_ids
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                country_s1.append(np.repeat(s1_ids[chunk], running_vals.shape[0]))
                country_cand.append(running_ids.T.cpu().numpy().ravel())
                country_score.append(running_vals.T.cpu().numpy().ravel())
                del q, running_vals, running_ids
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
    