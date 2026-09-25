"""Multilingual sentence embeddings of the raw (untransliterated) name + address; nearest neighbours by cosine.

See docs/blocking.md. Complements TfidfNgramBlocker: it reads native scripts directly, so a Hindi-script
"परफेक्ट मीडिया" can land near "Perfect Media", which transliteration ("prphekt midiya") destroys.
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PAIR_COLUMNS = ["s1_id", "cand_id", "score"]
MAX_SCORES = 250_000_000  # scores held at once per batch


class EmbeddingBlocker:
    def __init__(self, device: str = "cpu", model: str = "intfloat/multilingual-e5-small", batch_size: int = 512):
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = SentenceTransformer(model, device=str(self.device), model_kwargs={"torch_dtype": dtype})
        self.prefix = "query: " if "e5" in model.lower() else ""  # E5 models expect this prefix on inputs
        self.batch_size = batch_size

    def _encode(self, records: pd.DataFrame) -> torch.Tensor:
        text = self.prefix + records["raw_name"] + " | " + records["raw_address"]
        return self.model.encode(text.tolist(), batch_size=self.batch_size, convert_to_tensor=True,
                                 normalize_embeddings=True, show_progress_bar=False)

    def fit(self, targets: pd.DataFrame) -> "EmbeddingBlocker":
        """Encode the S2+S3 records; keep one embedding matrix per country."""
        vectors = self._encode(targets)
        ids = targets["entity_id"].to_numpy()
        self.targets = {country: (vectors[torch.from_numpy(rows).to(vectors.device)], ids[rows])
                        for country, rows in targets.groupby("country").indices.items()}
        return self

    def query(self, s1_records: pd.DataFrame, k: int) -> pd.DataFrame:
        """Top-k same-country targets per S1 by cosine similarity, best first."""
        vectors = self._encode(s1_records)
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
                best = torch.topk(vectors[torch.from_numpy(chunk).to(vectors.device)] @ targets.T, top_k, dim=1)
                frames.append(pd.DataFrame({
                    "s1_id": np.repeat(s1_ids[chunk], top_k),
                    "cand_id": target_ids[best.indices.cpu().numpy().ravel()],
                    "score": best.values.float().cpu().numpy().ravel(),
                }))
        if not frames:
            return pd.DataFrame(columns=PAIR_COLUMNS)
        return pd.concat(frames, ignore_index=True)
