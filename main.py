"""Entry point: wires the pipeline stages together. Configured by .env; swap an implementation by name there."""

import csv
import os
import time
from pathlib import Path

import pandas as pd

from er.blockers.embedding import EmbeddingBlocker
from er.blockers.tfidf import TfidfNgramBlocker
from er.blockers.union import UnionBlocker
from er.decision import decide, tune_threshold
from er.cache import RECORD_COLUMNS, PairCache
from er.evaluate import F05Evaluator
from er.features import FEATURE_COLUMNS, PairFeaturizer
from er.matcher import XgbMatcher
from er.normalize import RuleNormalizer
from er.output import write_candidate_pairs, write_matching_results
from er.split import HashSplitter
from er.training_pairs import build_training_pairs
from er.transliterate import AnyAsciiTransliterator

SPLITS_DIR = Path("data/splits")

TRANSLITERATORS = {"anyascii": AnyAsciiTransliterator}
NORMALIZERS = {"rules": RuleNormalizer}
BLOCKERS = {"tfidf": TfidfNgramBlocker, "embedding": EmbeddingBlocker}
MATCHERS = {"xgb": XgbMatcher}


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

    # MATCHER_MODEL: an existing file is loaded instead of training (TRAIN_SETS unused); a missing one is
    # trained on TRAIN_SETS and saved there, so the next run loads it.
    model_path = config["MATCHER_MODEL"]
    load_model = bool(model_path) and Path(model_path).exists()
    train_sets = [] if load_model else [int(k) for k in config["TRAIN_SETS"].split(",") if k.strip()]
    tune_sets = [int(k) for k in config["TUNE_SETS"].split(",") if k.strip()]
    if config["MATCHER"] and (not tune_sets or not (load_model or train_sets) or set(train_sets) & set(tune_sets)
                              or set(sets) & set(train_sets + tune_sets)):
        raise SystemExit("MATCHER needs TUNE_SETS and (unless MATCHER_MODEL exists) TRAIN_SETS, "
                         "disjoint from each other and from SETS")
    if any(not 0 <= k < n_sets for k in train_sets + tune_sets):
        raise SystemExit(f"TRAIN_SETS/TUNE_SETS in .env must be between 0 and N_SETS-1 ({n_sets - 1})")
    needed = sets + (train_sets + tune_sets if config["MATCHER"] else [])
    if needed and not all((splits_dir / f"set_{k}").exists() for k in needed):
        print(f"writing the {n_sets} training sets to {splits_dir}")
        HashSplitter(n_sets).write(data_dir / "train", splits_dir)

    def load_source(set_list: list[int], source: int) -> pd.DataFrame:
        if set_list:  # training sets from data/splits/, concatenated
            return pd.concat([read_tsv(splits_dir / f"set_{k}" / f"source{source}.tsv", nrows) for k in set_list],
                             ignore_index=True)
        return read_tsv(data_dir / split / f"{split}_source{source}.tsv", nrows)

    def prepare(raw: pd.DataFrame) -> pd.DataFrame:
        # The transliterator rewrites business_name/address; keep the originals for the embedding blocker.
        return normalizer.transform(transliterator.transform(raw)).assign(
            raw_name=raw["business_name"], raw_address=raw["business_address"])

    def block(set_list: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load, prepare and block the given sets (or the full SPLIT when empty): (s1, targets, pairs)."""
        sources = {s: prepare(load_source(set_list, s)) for s in (1, 2, 3)}
        print(f"sets {set_list or split}: " + ", ".join(f"source{s} {len(df):,}" for s, df in sources.items()))
        targets = pd.concat([sources[2], sources[3]], ignore_index=True)
        start = time.perf_counter()
        blocker.fit(targets)
        pairs = blocker.query(sources[1], block_k)
        print(f"  blocking ({config['BLOCKER']} on {blocker.device}): {len(pairs):,} candidate pairs "
              f"in {time.perf_counter() - start:.0f}s")
        return sources[1], targets, pairs

    def load_truth(set_list: list[int]) -> dict[str, set[str]]:
        truth = {}
        for k in set_list:
            truth |= evaluator.load(splits_dir / f"set_{k}" / "ground_truth.tsv")
        return truth

    featurizer = PairFeaturizer()
    cache_settings = {key: config[key] for key in ("TRANSLITERATOR", "NORMALIZER", "BLOCKER", "BLOCK_K", "EMBED_MODEL")}
    cache = (PairCache(Path(config["CACHE_DIR"]), {**cache_settings, "features": FEATURE_COLUMNS})
             if config["CACHE_DIR"] and nrows is None else None)

    def featurized(set_list: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """(pairs + features, S1 records, target records) for the given sets; cached per set list."""
        if cache and set_list and (hit := cache.load(set_list)) is not None:
            print(f"sets {set_list}: {len(hit[0]):,} featurized pairs loaded from {cache.dir}")
            return hit
        s1, targets, pairs = block(set_list)
        start = time.perf_counter()
        features = featurizer.transform(pairs, s1, targets)
        print(f"  featurized in {time.perf_counter() - start:.0f}s")
        if cache and set_list:
            cache.save(set_list, features, s1, targets)
        return features, s1[RECORD_COLUMNS], targets[RECORD_COLUMNS]

    evaluator = F05Evaluator()
    features, s1_records, _ = featurized(sets)
    truth = load_truth(sets) if sets and nrows is None else None  # complete ground truth only for full sets

    if truth:
        def report(label: str, subset: pd.DataFrame) -> None:
            candidates = subset.groupby("s1_id")["cand_id"].agg(set).to_dict()
            print(f"  {label}: recall {evaluator.blocking_recall(candidates, truth):.3f}  "
                  f"mean candidates: {evaluator.mean_candidates(candidates, truth):.1f}")

        for k in sorted({1, 5, 10, block_k}):
            report(f"top {k}", features[features["rank"] < k])
        report("all candidates", features)
        for member in [c.removeprefix("found_by_") for c in features.columns if c.startswith("found_by_")]:
            if member != "all":
                report(f"found by {member}", features[features[f"found_by_{member}"] == 1])

    if not config["MATCHER"]:
        return
    matcher = pick(MATCHERS, config, "MATCHER")
    s1_ids, candidates = s1_records["entity_id"], features[["s1_id", "cand_id"]]

    def labeled_features(set_list: list[int]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
        set_truth = load_truth(set_list)
        set_features, _, _ = featurized(set_list)
        return build_training_pairs(set_features, set_truth), set_truth

    if load_model:
        matcher = type(matcher).load(model_path)
        print(f"matcher ({config['MATCHER']}) loaded from {model_path}, not retrained")
    else:
        start = time.perf_counter()
        train_features, _ = labeled_features(train_sets)
        matcher.fit(train_features)
        print(f"matcher ({config['MATCHER']}) trained on sets {train_sets}: {len(train_features):,} pairs, "
              f"{int(train_features['label'].sum()):,} true, in {time.perf_counter() - start:.0f}s")
        del train_features
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            matcher.save(model_path)
            print(f"  saved to {model_path}")

    tune_features, tune_truth = labeled_features(tune_sets)
    threshold, tune_f05 = tune_threshold(matcher.predict(tune_features), tune_truth, evaluator)
    print(f"threshold {threshold:.2f} chosen on sets {tune_sets} (F0.5 there {tune_f05:.4f})")
    del tune_features

    matches = decide(matcher.predict(features), threshold)
    Path("output").mkdir(exist_ok=True)
    write_matching_results(matches, s1_ids, "output/matching_results.tsv")
    write_candidate_pairs(candidates, s1_ids, "output/candidate_pairs.tsv")
    print(f"wrote output/matching_results.tsv ({len(matches):,} matches) and output/candidate_pairs.tsv")

    if truth:
        predicted = matches.groupby("s1_id")["cand_id"].agg(set).to_dict()
        top1 = features[features["rank"] == 0].groupby("s1_id")["cand_id"].agg(set).to_dict()
        print(f"F0.5 on sets {sets}: {evaluator.score(predicted, truth):.4f}")
        print(f"  baseline, predict nothing:         {evaluator.score({}, truth):.4f}")
        print(f"  baseline, top blocker candidate:   {evaluator.score(top1, truth):.4f}")

if __name__ == "__main__":
    main()
