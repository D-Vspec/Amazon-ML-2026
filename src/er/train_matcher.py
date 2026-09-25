"""Train the matcher on one or more sets, validate on another, write outputs for inspection.

Run from the repo root:
    uv run python scripts/train_matcher.py --train-set 0 --val-set 1
    uv run python scripts/train_matcher.py --train-set 0 1 --val-set 2   # train on set_0 + set_1

What it does, in order, with timing and sanity numbers printed at each step:
1. Load data/splits/10_sets/set_<n> for every --train-set n and for --val-set (run main.py once
   first if these don't exist yet — HashSplitter creates them). Multiple train sets are concatenated;
   HashSplitter assigns each entity to exactly one set by id hash, so there's no overlap to worry
   about when combining them.
2. Transliterate + normalize S1/S2/S3 for train and val.
3. Block train and val independently (fit on that side's S2+S3, query with that side's S1, k=BLOCK_K).
4. Featurize and label the candidate pairs.
5. Train XGBoost on the TRAIN side only.
6. Score the VAL set: predict, tune the threshold against real F0.5, decide, write output files.

Fitting the matcher on one set and validating on another follows the same rule as the normalizer
docs: don't score on data you fit on. --val-set must not appear in --train-set.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from er.transliterate import AnyAsciiTransliterator
from er.normalize import RuleNormalizer
from er.blockers.embedding import EmbeddingBlocker
from er.blockers.tfidf import TfidfNgramBlocker
from er.blockers.union import UnionBlocker
from er.evaluate import F05Evaluator
from er.features import PairFeaturizer, FEATURE_COLUMNS
from er.training_pairs import build_training_pairs, summarize
from er.matcher import XgbMatcher
from er.decision import decide, tune_threshold
from er.output import write_matching_results, write_candidate_pairs

SPLITS_DIR = Path("data/splits/10_sets")
BLOCK_K = 20


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.label}] starting...", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"[{self.label}] done in {time.time() - self.t0:.1f}s", flush=True)


def load_set(set_dir: Path) -> dict[str, pd.DataFrame]:
    read = lambda f: pd.read_csv(set_dir / f, sep="\t", dtype=str, keep_default_na=False)
    return {"s1": read("source1.tsv"), "s2": read("source2.tsv"), "s3": read("source3.tsv")}


def load_sets(set_nums: list[int], evaluator: F05Evaluator) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load and concatenate multiple set_<n> directories. Safe to combine: HashSplitter assigns
    each entity to exactly one set by id hash, so there's no overlap between sets' rows or ids."""
    raws = [load_set(SPLITS_DIR / f"set_{n}") for n in set_nums]
    combined = {
        "s1": pd.concat([r["s1"] for r in raws], ignore_index=True),
        "s2": pd.concat([r["s2"] for r in raws], ignore_index=True),
        "s3": pd.concat([r["s3"] for r in raws], ignore_index=True),
    }
    truth = {}
    for n in set_nums:
        truth.update(evaluator.load(SPLITS_DIR / f"set_{n}" / "ground_truth.tsv"))
    return combined, truth


def prepare(raw: dict[str, pd.DataFrame], translit, norm) -> dict[str, pd.DataFrame]:
    """Transliterate + normalize S1/S2/S3; return s1 and the concatenated s2+s3 target frame."""
    # raw_name/raw_address: the untransliterated text the embedding blocker reads (docs/blocking.md).
    frames = {k: norm.transform(translit.transform(df)).assign(raw_name=df["business_name"],
                                                               raw_address=df["business_address"])
              for k, df in raw.items()}
    target = pd.concat([frames["s2"], frames["s3"]], ignore_index=True)
    return {"s1": frames["s1"], "target": target}


def block_and_featurize(prepared: dict[str, pd.DataFrame], device: str) -> pd.DataFrame:
    # TF-IDF and the multilingual embedder in parallel; recall 0.990 vs 0.977 for TF-IDF alone (docs/blocking.md).
    blocker = UnionBlocker({"tfidf": TfidfNgramBlocker(device=device), "embedding": EmbeddingBlocker(device=device)})
    blocker.fit(prepared["target"])
    candidates = blocker.query(prepared["s1"], k=BLOCK_K)
    feats = PairFeaturizer().transform(candidates, prepared["s1"], prepared["target"])
    return candidates, feats


def run(train_sets: list[int], val_set: int, device: str):
    if val_set in train_sets:
        raise ValueError(f"--val-set {val_set} also appears in --train-set {train_sets}; "
                         "validating on data trained on defeats the point")

    evaluator = F05Evaluator()
    translit, norm = AnyAsciiTransliterator(), RuleNormalizer()

    train_label = "+".join(f"set_{n}" for n in train_sets)
    with Timer(f"load {train_label}"):
        train_raw, train_truth = load_sets(train_sets, evaluator)
    print(f"  S1={len(train_raw['s1'])}  S2={len(train_raw['s2'])}  S3={len(train_raw['s3'])}"
          f"  singleton_rate={sum(1 for v in train_truth.values() if not v) / len(train_truth):.4f}")

    with Timer(f"load set_{val_set}"):
        val_raw = load_set(SPLITS_DIR / f"set_{val_set}")
        val_truth = evaluator.load(SPLITS_DIR / f"set_{val_set}" / "ground_truth.tsv")
    print(f"  S1={len(val_raw['s1'])}  S2={len(val_raw['s2'])}  S3={len(val_raw['s3'])}"
          f"  singleton_rate={sum(1 for v in val_truth.values() if not v) / len(val_truth):.4f}")

    with Timer("normalize train"):
        train_prep = prepare(train_raw, translit, norm)
    with Timer("normalize val"):
        val_prep = prepare(val_raw, translit, norm)

    with Timer(f"block+featurize train (k={BLOCK_K}, device={device})"):
        train_cand, train_feats = block_and_featurize(train_prep, device)
    train_recall = evaluator.blocking_recall({s1: set(g.cand_id) for s1, g in train_cand.groupby("s1_id")}, train_truth)
    print(f"  candidate pairs={len(train_cand)}  blocking_recall@{BLOCK_K}={train_recall:.4f}"
          f"  mean_candidates={len(train_cand)/len(train_raw['s1']):.1f}")
    if train_recall < 0.85:
        print("  !! blocking recall looks low for this data size — check normalization/country labels"
              " before trusting the matcher numbers below.")

    with Timer(f"block+featurize val (k={BLOCK_K}, device={device})"):
        val_cand, val_feats = block_and_featurize(val_prep, device)
    val_recall = evaluator.blocking_recall({s1: set(g.cand_id) for s1, g in val_cand.groupby("s1_id")}, val_truth)
    print(f"  candidate pairs={len(val_cand)}  blocking_recall@{BLOCK_K}={val_recall:.4f}")

    with Timer("label"):
        train_labeled = train_feats.merge(
            build_training_pairs(train_cand, train_truth)[["s1_id", "cand_id", "label"]],
            on=["s1_id", "cand_id"])
        val_labeled = val_feats.merge(
            build_training_pairs(val_cand, val_truth)[["s1_id", "cand_id", "label"]],
            on=["s1_id", "cand_id"])
    print(f"  train: {summarize(train_labeled, train_truth)}")

    with Timer("train xgboost"):
        matcher = XgbMatcher().fit(train_labeled)
        matcher.save("model.json")
        print("  saved model to model.json")
    print("  feature importance:")
    print(matcher.feature_importance().to_string())

    with Timer("score val + tune threshold"):
        scored = matcher.predict(val_labeled)
        best_t, best_f = tune_threshold(scored, val_truth, evaluator)
    baseline_f = evaluator.score({}, val_truth)  # predict nothing for everyone = singleton rate
    top1_pred = {s1: {g.sort_values("score", ascending=False).cand_id.iloc[0]}
                for s1, g in val_cand.groupby("s1_id")}
    top1_f = evaluator.score(top1_pred, val_truth)
    print(f"  threshold={best_t:.2f}  F0.5={best_f:.4f}")
    print(f"  baseline, predict nothing:        F0.5={baseline_f:.4f}")
    print(f"  baseline, always take top-1 cand: F0.5={top1_f:.4f}")
    if best_f <= baseline_f + 0.02:
        print("  !! trained matcher barely beats predicting nothing — check labels/features before"
              " trusting this, something upstream is likely broken.")

    with Timer("write outputs"):
        final = decide(scored, best_t)
        Path("output").mkdir(exist_ok=True)
        write_matching_results(final, val_raw["s1"]["entity_id"], "output/matching_results.tsv")
        write_candidate_pairs(val_cand, val_raw["s1"]["entity_id"], "output/candidate_pairs.tsv")
    print("  wrote output/matching_results.tsv and output/candidate_pairs.tsv"
          " — run utils/validate_submission.py on these before trusting the format.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", type=int, nargs="+", default=[0],
                    help="One or more set numbers to train on, e.g. --train-set 0 1")
    ap.add_argument("--val-set", type=int, default=1)
    ap.add_argument("--device", default="cuda")  # falls back to cpu automatically if no GPU
    args = ap.parse_args()
    run(args.train_set, args.val_set, args.device)