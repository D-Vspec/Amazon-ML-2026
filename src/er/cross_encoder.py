"""Fine-tuned multilingual cross-encoder: reads both raw records together and scores P(same business).

See docs/matching.md. Its probability is added as the `score_cross_encoder` column, which PairFeaturizer
passes through and XgbMatcher trains on, so XGBoost learns when to trust it (stacking).
"""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

# Models whose Hub config needs trust_remote_code, pinned to the code commit that was reviewed
# (Alibaba-NLP/new-impl: only torch/transformers maths, no I/O or network; see docs/matching.md).
REVIEWED_REMOTE_CODE = {"Alibaba-NLP/gte-multilingual-reranker-base": "40ced75c3017eb27626c9d4ea981bde21a2662f4"}
PREDICT_CHUNK = 500_000  # pairs turned into text and scored at once


class CrossEncoderScorer:
    def __init__(self, device: str = "cpu", model: str = "Alibaba-NLP/gte-multilingual-reranker-base",
                 max_length: int = 128, batch_size: int = 64, train_pairs: int = 1_000_000, seed: int = 0,
                 path: str | None = None):
        """`model` is the Hub base checkpoint; `path` loads a fine-tuned copy of it instead (see load())."""
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.base_model, self.max_length, self.batch_size = model, max_length, batch_size
        self.train_pairs, self.seed = train_pairs, seed
        self.model = self._load(path or model)

    def _load(self, path: str) -> CrossEncoder:
        code_revision = REVIEWED_REMOTE_CODE.get(self.base_model)
        extra = {"code_revision": code_revision} if code_revision else {}
        dtype = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        # Sigmoid always: some checkpoints (e.g. mmarco-mMiniLMv2) save no activation and return raw logits.
        return CrossEncoder(path, num_labels=1, max_length=self.max_length, device=self.device,
                            trust_remote_code=code_revision is not None, activation_fn=torch.nn.Sigmoid(),
                            config_kwargs=extra, model_kwargs={**extra, **dtype})

    @staticmethod
    def _texts(pairs: pd.DataFrame, s1_records: pd.DataFrame, targets: pd.DataFrame) -> tuple[list, list]:
        def text(records: pd.DataFrame, ids: pd.Series) -> list[str]:
            joined = records.set_index("entity_id")
            return (joined["raw_name"] + " | " + joined["raw_address"]).reindex(ids).fillna("").tolist()
        return text(s1_records, pairs["s1_id"]), text(targets, pairs["cand_id"])

    def fit(self, labeled: pd.DataFrame, s1_records: pd.DataFrame, targets: pd.DataFrame) -> "CrossEncoderScorer":
        """Fine-tune with binary cross-entropy on all positives plus the hardest negatives (highest score first)."""
        positives = labeled[labeled["label"] == 1]
        negatives = labeled[labeled["label"] == 0].nlargest(max(0, self.train_pairs - len(positives)), "score")
        sample = pd.concat([positives, negatives]).sample(frac=1, random_state=self.seed)
        left, right = self._texts(sample, s1_records, targets)
        swap = np.random.default_rng(self.seed).random(len(sample)) < 0.5  # a match is a match in either order
        first = [b if s else a for a, b, s in zip(left, right, swap)]
        second = [a if s else b for a, b, s in zip(left, right, swap)]
        data = Dataset.from_dict({"text1": first, "text2": second, "label": sample["label"].astype(float).tolist()})
        args = CrossEncoderTrainingArguments(
            output_dir="models/.ce_checkpoints", num_train_epochs=1, per_device_train_batch_size=self.batch_size,
            learning_rate=2e-5, warmup_ratio=0.1, fp16=self.device == "cuda", save_strategy="no",
            logging_steps=500, report_to="none", seed=self.seed)
        if self.device == "cuda":  # train in fp32 weights with fp16 autocast; inference stays fp16
            self.model.model.float()
        CrossEncoderTrainer(model=self.model, args=args, train_dataset=data,
                            loss=BinaryCrossEntropyLoss(self.model)).train()
        if self.device == "cuda":
            self.model.model.half()
        return self

    def score(self, pairs: pd.DataFrame, s1_records: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
        """P(match) for every pair, in pairs' order."""
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), PREDICT_CHUNK):
            chunk = pairs.iloc[start:start + PREDICT_CHUNK]
            left, right = self._texts(chunk, s1_records, targets)
            out[start:start + len(chunk)] = self.model.predict(list(zip(left, right)), batch_size=self.batch_size * 4,
                                                               show_progress_bar=False)
        return out

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", model: str = "Alibaba-NLP/gte-multilingual-reranker-base",
             **kwargs) -> "CrossEncoderScorer":
        """Load a fine-tuned model from `path`; `model` names its base checkpoint (for the remote-code pin)."""
        return cls(device, model, path=path, **kwargs)
