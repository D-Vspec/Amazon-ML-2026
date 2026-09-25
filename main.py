"""Entry point: wires the pipeline stages together. Configured by .env; swap an implementation by name there."""

import csv
import os
import time
from pathlib import Path

import pandas as pd

from er.blockers.embedding import EmbeddingBlocker
from er.blockers.tfidf import TfidfNgramBlocker
from er.blockers.union import UnionBlocker
from er.evaluate import F05Evaluator
from er.normalize import RuleNormalizer
from er.split import HashSplitter
from er.transliterate import AnyAsciiTransliterator

SPLITS_DIR = Path("data/splits")

TRANSLITERATORS = {"anyascii": AnyAsciiTransliterator}
NORMALIZERS = {"rules": RuleNormalizer}
BLOCKERS = {"tfidf": TfidfNgramBlocker, "embedding": EmbeddingBlocker}


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


def pick(registry: dict, config: dict[str, str], key: str):
    name = config[key]
    if name not in registry:
        raise SystemExit(f"{key}={name!r} in .env is not one of {sorted(registry)}")
    return registry[name]()


def make_blocker(config: dict[str, str]):
    """BLOCKER=tfidf, =embedding, or =tfidf+embedding (a union of the listed blockers)."""
    names = config["BLOCKER"].split("+")
    unknown = [name for name in names if name not in BLOCKERS]
    if unknown:
        raise SystemExit(f"BLOCKER={config['BLOCKER']!r} in .env: {unknown} not in {sorted(BLOCKERS)}")
    members = {name: BLOCKERS[name](device=config["DEVICE"], **({"model": config["EMBED_MODEL"]}
                                                                 if name == "embedding" else {}))
               for name in names}
    return members[names[0]] if len(names) == 1 else UnionBlocker(members)


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
    blocker = make_blocker(config)
    block_k = int(config["BLOCK_K"])

    if sets and not all((splits_dir / f"set_{k}").exists() for k in sets):
        print(f"writing the {n_sets} training sets to {splits_dir}")
        HashSplitter(n_sets).write(data_dir / "train", splits_dir)

    def load_source(source: int) -> pd.DataFrame:
        if sets:  # training sets from data/splits/, concatenated
            return pd.concat([read_tsv(splits_dir / f"set_{k}" / f"source{source}.tsv", nrows) for k in sets],
                             ignore_index=True)
        return read_tsv(data_dir / split / f"{split}_source{source}.tsv", nrows)

    def prepare(raw: pd.DataFrame) -> pd.DataFrame:
        # The transliterator rewrites business_name/address; keep the originals for the embedding blocker.
        return normalizer.transform(transliterator.transform(raw)).assign(
            raw_name=raw["business_name"], raw_address=raw["business_address"])

    sources = {s: prepare(load_source(s)) for s in (1, 2, 3)}

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
        def report(label: str, subset: pd.DataFrame) -> None:
            candidates = subset.groupby("s1_id")["cand_id"].agg(set).to_dict()
            print(f"  {label}: recall {evaluator.blocking_recall(candidates, truth):.3f}  "
                  f"mean candidates: {evaluator.mean_candidates(candidates, truth):.1f}")

        rank = pairs.groupby("s1_id").cumcount()
        for k in sorted({1, 5, 10, block_k}):
            report(f"top {k}", pairs[rank < k])
        report("all candidates", pairs)
        for member in [c.removeprefix("score_") for c in pairs.columns if c.startswith("score_")]:
            report(f"found by {member}", pairs[pairs[f"score_{member}"] > 0])


if __name__ == "__main__":
    main()
