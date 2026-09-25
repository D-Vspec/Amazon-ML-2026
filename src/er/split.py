"""Split the training data into 10 equal, cluster-aware sets. See docs/data_splits.md."""

import zlib
from contextlib import ExitStack
from pathlib import Path


class HashSplitter:
    n_sets = 10

    def set_of(self, entity_id: str) -> int:
        return zlib.crc32(entity_id.encode()) % self.n_sets

    def write(self, train_dir: Path, out_dir: Path) -> dict[int, dict[str, int]]:
        """Write `<out_dir>/set_<k>/{source1,source2,source3,ground_truth}.tsv` for k in 0..9.

        An S1 record and all its matches go to the S1's set; unmatched S2/S3 records (distractors)
        use their own id. Returns row counts per set and file.
        """
        # Every matched S2/S3 id -> its S1's set. Each id has at most one owner.
        owner_set = {}
        with open(train_dir / "train_ground_truth.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, ids = line.rstrip("\n").partition("\t")
                k = self.set_of(s1)
                for x in ids.split(","):
                    if x:
                        owner_set[x] = k

        counts = {k: {} for k in range(self.n_sets)}
        files = [("train_source1.tsv", "source1.tsv"), ("train_source2.tsv", "source2.tsv"),
                 ("train_source3.tsv", "source3.tsv"), ("train_ground_truth.tsv", "ground_truth.tsv")]
        for src_name, dst_name in files:
            with ExitStack() as stack, open(train_dir / src_name, encoding="utf-8") as src:
                header = next(src)
                outs = []
                for k in range(self.n_sets):
                    (out_dir / f"set_{k}").mkdir(parents=True, exist_ok=True)
                    outs.append(stack.enter_context(open(out_dir / f"set_{k}" / dst_name, "w", encoding="utf-8")))
                    outs[k].write(header)
                    counts[k][dst_name] = 0
                for line in src:
                    entity_id = line.split("\t", 1)[0]
                    k = owner_set.get(entity_id)
                    if k is None:  # an S1 record, a ground-truth row, or a distractor
                        k = self.set_of(entity_id)
                    outs[k].write(line)
                    counts[k][dst_name] += 1
        return counts
