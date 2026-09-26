"""Break down exactly where F0.5 is being lost on a validation set: false positives, false negatives,
and the entities the blocker never gave a chance. Run this before tuning anything blind.

Usage:
    uv run python scripts/analyze_errors.py --model model.json --val-set 2 --threshold 0.75

Needs the same set-loading / normalize / block / featurize helpers as train_matcher.py.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from er.transliterate import AnyAsciiTransliterator
from er.normalize import RuleNormalizer
from er.blockers.tfidf import TfidfNgramBlocker
from er.evaluate import F05Evaluator
from er.features import PairFeaturizer
from er.training_pairs import build_training_pairs
from er.matcher import XgbMatcher
from er.decision import decide

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


def prepare(raw, translit, norm):
    frames = {k: norm.transform(translit.transform(df)) for k, df in raw.items()}
    target = pd.concat([frames["s2"], frames["s3"]], ignore_index=True)
    return {"s1": frames["s1"], "target": target}


def run(model_path: str, val_set: int, threshold: float, device: str, n_examples: int):
    evaluator = F05Evaluator()
    translit, norm = AnyAsciiTransliterator(), RuleNormalizer()

    with Timer(f"load set_{val_set}"):
        raw = load_set(SPLITS_DIR / f"set_{val_set}")
        truth = evaluator.load(SPLITS_DIR / f"set_{val_set}" / "ground_truth.tsv")
    print(f"  S1={len(raw['s1'])}  S2={len(raw['s2'])}  S3={len(raw['s3'])}")

    with Timer("normalize"):
        prep = prepare(raw, translit, norm)

    with Timer(f"block (k={BLOCK_K}, device={device})"):
        blocker = TfidfNgramBlocker(device=device)
        blocker.fit(prep["target"])
        candidates = blocker.query(prep["s1"], k=BLOCK_K)
    print(f"  candidate pairs={len(candidates)}")

    with Timer("featurize + label"):
        feats = PairFeaturizer().transform(candidates, prep["s1"], prep["target"])
        labeled = feats.merge(build_training_pairs(candidates, truth)[["s1_id", "cand_id", "label"]],
                              on=["s1_id", "cand_id"])
    print(f"  positives={int(labeled.label.sum())}  negatives={int((labeled.label==0).sum())}")

    with Timer("load model + predict"):
        matcher = XgbMatcher()
        matcher.model = xgb.XGBClassifier()
        matcher.model.load_model(model_path)
        scored = matcher.predict(labeled)

    with Timer("decide"):
        final = decide(scored, threshold)
        pred = {s1: set(g.cand_id) for s1, g in final.groupby("s1_id")}
    print("  computing per-entity breakdown...")

    # ---- per-entity scores, worst first ----
    rows = []
    for s1, t in truth.items():
        p = pred.get(s1, set())
        f = evaluator.entity_score(p, t)
        rows.append({"s1_id": s1, "f0_5": f, "n_true": len(t), "n_pred": len(p),
                     "false_pos": len(p - t), "false_neg": len(t - p)})
    scores = pd.DataFrame(rows)
    print("=" * 70)
    print(f"OVERALL: macro F0.5 = {evaluator.score(pred, truth):.4f}   n_entities={len(scores)}")
    print(f"  perfect (F0.5=1.0): {(scores.f0_5 == 1.0).mean():.1%}")
    print(f"  zero (F0.5=0.0):    {(scores.f0_5 == 0.0).mean():.1%}")
    print(f"  partial credit:     {((scores.f0_5 > 0) & (scores.f0_5 < 1.0)).mean():.1%}")

    # ---- how much comes from blocking (can never be fixed by the matcher) vs the matcher itself ----
    cand_by_s1 = {s1: set(g.cand_id) for s1, g in candidates.groupby("s1_id")}
    unreachable = {s1: len(t - cand_by_s1.get(s1, set())) for s1, t in truth.items() if t}
    fully_unreachable = sum(1 for s1, n in unreachable.items() if n == len(truth[s1]))
    partially_unreachable = sum(1 for s1, n in unreachable.items() if 0 < n < len(truth[s1]))
    print(f"\nBLOCKING CEILING (can't be fixed by tuning the matcher):")
    print(f"  entities where ALL true matches were missed by blocking: {fully_unreachable}")
    print(f"  entities where SOME true matches were missed by blocking: {partially_unreachable}")
    print(f"  -> if these are a big share of your 'zero'/'partial credit' rows above,")
    print(f"     fix blocking (word table / second blocker) before touching the matcher.")

    # ---- false positives: what does the matcher wrongly say yes to? ----
    # Split into two very different failure modes:
    #  - "stolen": cand_id is some OTHER s1's true match, taken by this s1 instead. A real bug in
    #    the assignment strategy (independent per-pair thresholding lets one candidate be claimed
    #    by multiple S1s) -- fixable with global 1-to-1 assignment, not more features.
    #  - "unclaimed": cand_id belongs to no one in ground truth at all. Two textually-near-identical
    #    but legally distinct entities (shared address, sibling business, etc) -- no assignment fix
    #    helps here; it's an irreducible ambiguity unless there's an out-of-band signal to use.
    owner_of = {cid: s1 for s1, ids in truth.items() for cid in ids}

    fp = scored.merge(final[["s1_id", "cand_id"]].assign(kept=1), on=["s1_id", "cand_id"], how="left")
    fp = fp[(fp.kept == 1) & (fp.label == 0)].sort_values("match_proba", ascending=False)
    fp["true_owner"] = fp["cand_id"].map(owner_of)
    fp["case"] = fp["true_owner"].apply(lambda o: "stolen" if pd.notna(o) else "unclaimed")

    print(f"\nFALSE POSITIVES: {len(fp)} pairs wrongly kept")
    case_counts = fp["case"].value_counts()
    n_stolen = int(case_counts.get("stolen", 0))
    n_unclaimed = int(case_counts.get("unclaimed", 0))
    print(f"  stolen (belongs to a different S1 in ground truth): {n_stolen}  "
          f"({n_stolen / len(fp):.1%} of false positives)" if len(fp) else "")
    print(f"  unclaimed (no true owner -- genuine near-duplicate): {n_unclaimed}  "
          f"({n_unclaimed / len(fp):.1%} of false positives)" if len(fp) else "")
    if len(fp):
        print("  -> if 'stolen' dominates, fix decide() with global 1-to-1 assignment before anything else.")
        print("  -> if 'unclaimed' dominates, this is a data-ambiguity ceiling; more features/assignment")
        print("     logic won't fix it without an out-of-band signal (e.g. a registration number).")
    print(fp[["s1_id", "cand_id", "case", "true_owner", "match_proba", "score", "name_jaccard", "addr_jaccard"]]
          .head(n_examples).to_string(index=False))

    # ---- false negatives that WERE in the candidate list but scored too low ----
    fn = scored[(scored.label == 1) & (scored.match_proba < threshold)].sort_values("match_proba", ascending=False)
    print(f"\nFALSE NEGATIVES (in candidates, but scored below threshold): {len(fn)} pairs")
    print(fn[["s1_id", "cand_id", "match_proba", "score", "name_jaccard", "addr_jaccard"]].head(n_examples).to_string(index=False))

    # ---- show the actual text for the worst false positives and false negatives ----
    s1_txt = raw["s1"].set_index("entity_id")[["business_name", "business_address", "country"]]
    t_txt = pd.concat([raw["s2"], raw["s3"]]).set_index("entity_id")[["business_name", "business_address"]]
    print(f"\nWORST FALSE POSITIVES (raw text):")
    for _, r in fp.head(10).iterrows():
        s = s1_txt.loc[r.s1_id]
        c = t_txt.loc[r.cand_id]
        tag = f"STOLEN from {r.true_owner}" if r.case == "stolen" else "unclaimed"
        print(f"  [{r.match_proba:.2f}] ({tag}) S1: {s.business_name} | {s.business_address} ({s.country})")
        print(f"          cand: {c.business_name} | {c.business_address}")
    print(f"\nWORST FALSE NEGATIVES (raw text):")
    for _, r in fn.head(10).iterrows():
        s = s1_txt.loc[r.s1_id]
        c = t_txt.loc[r.cand_id]
        print(f"  [{r.match_proba:.2f}] S1: {s.business_name} | {s.business_address} ({s.country})")
        print(f"          cand: {c.business_name} | {c.business_address}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.json")
    ap.add_argument("--val-set", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=15, dest="n_examples")
    args = ap.parse_args()
    run(args.model, args.val_set, args.threshold, args.device, args.n_examples)