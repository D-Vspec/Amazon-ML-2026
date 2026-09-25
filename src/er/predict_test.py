"""Run the trained matcher on the real competition test set and write a submission.

Run from the repo root, after a model has been trained and saved (see train_matcher.py / matcher.py):
    uv run python src/er/predict_test.py --model model.json --test-dir "<path to dataset/test>" --threshold 0.85

Unlike train_matcher.py, there is no ground truth here, so there's no recall/F0.5 check and no
threshold tuning -- the threshold is passed in (use the value train_matcher.py printed, e.g. 0.85).

Writes output/matching_results.tsv and output/candidate_pairs.tsv covering every S1 entity_id in the
test set (empty string = no match), in the exact format validate_submission.py checks.
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
from er.features import PairFeaturizer
from er.matcher import XgbMatcher
from er.decision import decide
from er.output import write_matching_results, write_candidate_pairs

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


def load_test(test_dir: Path) -> dict[str, pd.DataFrame]:
    read = lambda f: pd.read_csv(test_dir / f, sep="\t", dtype=str, keep_default_na=False)
    return {"s1": read("test_source1.tsv"), "s2": read("test_source2.tsv"), "s3": read("test_source3.tsv")}


def prepare(raw: dict[str, pd.DataFrame], translit, norm) -> dict[str, pd.DataFrame]:
    # raw_name/raw_address: the untransliterated text the embedding blocker reads (docs/blocking.md).
    frames = {k: norm.transform(translit.transform(df)).assign(raw_name=df["business_name"],
                                                               raw_address=df["business_address"])
              for k, df in raw.items()}
    target = pd.concat([frames["s2"], frames["s3"]], ignore_index=True)
    return {"s1": frames["s1"], "target": target}


def run(model_path: str, test_dir: str, threshold: float, device: str):
    test_dir = Path(test_dir)
    translit, norm = AnyAsciiTransliterator(), RuleNormalizer()

    with Timer("load test"):
        raw = load_test(test_dir)
    print(f"  S1={len(raw['s1'])}  S2={len(raw['s2'])}  S3={len(raw['s3'])}")

    with Timer("normalize test"):
        prep = prepare(raw, translit, norm)

    with Timer(f"block+featurize test (k={BLOCK_K}, device={device})"):
        # Must match the blocker the model was trained with (train_matcher.py): TF-IDF + embedder in parallel.
        blocker = UnionBlocker({"tfidf": TfidfNgramBlocker(device=device),
                                "embedding": EmbeddingBlocker(device=device)})
        blocker.fit(prep["target"])
        candidates = blocker.query(prep["s1"], k=BLOCK_K)
        feats = PairFeaturizer().transform(candidates, prep["s1"], prep["target"])
    print(f"  candidate pairs={len(candidates)}  mean_candidates={len(candidates)/len(raw['s1']):.1f}")

    with Timer("load model + predict"):
        matcher = XgbMatcher.load(model_path)
        scored = matcher.predict(feats)

    with Timer("write outputs"):
        final = decide(scored, threshold)
        Path("output").mkdir(exist_ok=True)
        write_matching_results(final, raw["s1"]["entity_id"], "output/matching_results.tsv")
        write_candidate_pairs(candidates, raw["s1"]["entity_id"], "output/candidate_pairs.tsv")
    print("  wrote output/matching_results.tsv and output/candidate_pairs.tsv"
          " -- run utils/validate_submission.py --test-dir <test_dir> on these next.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a model saved by XgbMatcher.save(), e.g. model.json")
    ap.add_argument("--test-dir", required=True, help="Path to the real dataset/test directory")
    ap.add_argument("--threshold", type=float, required=True, help="Decision threshold, e.g. from train_matcher.py's tuning output")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.model, args.test_dir, args.threshold, args.device)