"""Entry point: wires the pipeline stages together. Configured by .env; swap an implementation by name there."""

import csv
import os
import time
from pathlib import Path

import pandas as pd

from er.blockers.tfidf import TfidfNgramBlocker
from er.evaluate import F05Evaluator
from er.normalize import RuleNormalizer
from er.split import HashSplitter
from er.transliterate import AnyAsciiTransliterator

SPLITS_DIR = Path("data/splits")

TRANSLITERATORS = {"anyascii": AnyAsciiTransliterator}
NORMALIZERS = {"rules": RuleNormalizer}
BLOCKERS = {"tfidf": TfidfNgramBlocker}


def load_config(path: Path = Path(".env")) -> dict[str, str]:
    """KEY=VALUE lines from `path` (blank lines and # comments skipped); environment variables override.

    Falls back to the committed `.env.example` when there is no local `.env`.
    """
    if not path.exists():
        path = path.with_name(".env.example")
    config = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return {key: os.environ.get(key, value) for key, value in config.items()}


def pick(registry: dict, config: dict[str, str], key: str, **kwargs):
    name = config[key]
    if name not in registry:
        raise SystemExit(f"{key}={name!r} in .env is not one of {sorted(registry)}")
    return registry[name](**kwargs)


def read_tsv(path: Path, nrows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=nrows)


def main():
    config = load_config()
    data_dir, split = Path(config["DATA_DIR"]), config["SPLIT"]
    n_sets = int(config["N_SETS"])
    sets = [int(k) for k in config["SETS"].split(",") if k.strip()]
    if any(not 0 <= k < n_sets for k in sets):
        raise SystemExit(f"SETS={config['SETS']!r} in .env must be between 0 and N_SETS-1 ({n_sets - 1})")
    splits_dir = SPLITS_DIR / f"{n_sets}_sets"  # a different N assigns records differently
    nrows = int(config["NROWS"]) if config["NROWS"] else None
    transliterator = pick(TRANSLITERATORS, config, "TRANSLITERATOR")
    normalizer = pick(NORMALIZERS, config, "NORMALIZER")
    blocker = pick(BLOCKERS, config, "BLOCKER", device=config["DEVICE"])
    block_k = int(config["BLOCK_K"])

    if sets and not all((splits_dir / f"set_{k}").exists() for k in sets):
        print(f"writing the {n_sets} training sets to {splits_dir}")
        HashSplitter(n_sets).write(data_dir / "train", splits_dir)

    def load_source(source: int) -> pd.DataFrame:
        if sets:  # training sets from data/splits/, concatenated
            return pd.concat([read_tsv(splits_dir / f"set_{k}" / f"source{source}.tsv", nrows) for k in sets],
                             ignore_index=True)
        return read_tsv(data_dir / split / f"{split}_source{source}.tsv", nrows)

    sources = {s: normalizer.transform(transliterator.transform(load_source(s))) for s in (1, 2, 3)}

    for s, df in sources.items():
        print(f"source{s}: {len(df):,} rows")

    start = time.perf_counter()
    blocker.fit(pd.concat([sources[2], sources[3]], ignore_index=True))
    pairs = blocker.query(sources[1], block_k)
    print(f"blocking ({config['BLOCKER']} on {blocker.device}): {len(pairs):,} candidate pairs "
          f"in {time.perf_counter() - start:.0f}s")

    if sets and nrows is None:  # full sets: ground truth is complete, so blocking recall is meaningful
        evaluator = F05Evaluator()
        truth = {}
        for k in sets:
            truth |= evaluator.load(splits_dir / f"set_{k}" / "ground_truth.tsv")
        rank = pairs.groupby("s1_id").cumcount()
        for k in sorted({1, 5, 10, block_k}):
            candidates = pairs[rank < k].groupby("s1_id")["cand_id"].agg(set).to_dict()
            print(f"  recall@{k}: {evaluator.blocking_recall(candidates, truth):.3f}  "
                  f"mean candidates: {evaluator.mean_candidates(candidates, truth):.1f}")


if __name__ == "__main__":
    main()
