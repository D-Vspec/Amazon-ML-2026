"""Deterministic, cluster-aware subsets of the training data. See docs/data_splits.md."""

import zlib
from contextlib import ExitStack
from pathlib import Path

GROUND_TRUTH = "ground_truth.tsv"


class HashSplitter:
    n_buckets = 100
    # Validation tiers are nested (val_1 in val_5 in val_10); training never touches buckets 0-9.
    tier_buckets = {
        "val_1": range(0, 1),
        "val_5": range(0, 5),
        "val_10": range(0, 10),
        "train_10": range(10, 20),
        "train": range(10, 100),
    }

    def bucket(self, entity_id: str) -> int:
        return zlib.crc32(entity_id.encode()) % self.n_buckets

    def write(self, train_dir: Path, out_dir: Path, tiers: list[str] | None = None) -> dict[str, dict[str, int]]:
        """Write `<out_dir>/<tier>/{source1,source2,source3,ground_truth}.tsv` for each tier.

        An S1 record and all its matches share the S1's bucket; unmatched S2/S3 records (distractors)
        use their own id's bucket. Returns row counts per tier and file.
        """
        tiers = tiers or list(self.tier_buckets)
        tiers_of = [[t for t in tiers if b in self.tier_buckets[t]] for b in range(self.n_buckets)]

        # Every matched S2/S3 id -> its S1's bucket. Each id has at most one owner.
        owner_bucket = {}
        with open(train_dir / "train_ground_truth.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, ids = line.rstrip("\n").partition("\t")
                b = self.bucket(s1)
                for x in ids.split(","):
                    if x:
                        owner_bucket[x] = b

        counts = {t: {} for t in tiers}
        files = [("train_source1.tsv", "source1.tsv"), ("train_source2.tsv", "source2.tsv"),
                 ("train_source3.tsv", "source3.tsv"), ("train_ground_truth.tsv", GROUND_TRUTH)]
        for src_name, dst_name in files:
            with ExitStack() as stack, open(train_dir / src_name, encoding="utf-8") as src:
                header = next(src)
                outs = {}
                for t in tiers:
                    (out_dir / t).mkdir(parents=True, exist_ok=True)
                    outs[t] = stack.enter_context(open(out_dir / t / dst_name, "w", encoding="utf-8"))
                    outs[t].write(header)
                    counts[t][dst_name] = 0
                for line in src:
                    entity_id = line.split("\t", 1)[0]
                    b = owner_bucket.get(entity_id)
                    if b is None:  # an S1 record, a ground-truth row, or a distractor
                        b = self.bucket(entity_id)
                    for t in tiers_of[b]:
                        outs[t].write(line)
                        counts[t][dst_name] += 1
        return counts
