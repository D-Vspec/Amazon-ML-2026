# Data splits

Small, fixed, representative subsets of the training data. Use them to iterate in seconds, compare models in minutes, and keep a validation set that training never sees.

Code: `src/er/split.py` (`HashSplitter`). Tests: `tests/test_split.py`. Scoring on a split: [evaluation.md](evaluation.md).

## Why

The training data is 2,206,821 S1 records and 10,320,219 S2/S3 records. Normalizing it all takes about 7 minutes, and blocking plus matching will take much longer. There are no labels for the test set, so every local score has to come from held-out training data.

A naive random sample of *rows* would break the problem:
- If you sample S1 and S2/S3 rows independently, most true matches of a sampled S1 are missing, so recall looks terrible for no reason.
- If you sample only matched records, there's nothing wrong to match against, so precision looks perfect for no reason.

## What the ground truth tells us

Measured on `train_ground_truth.tsv` and the train source files:

| Fact | Value | Consequence |
|---|---|---|
| S2/S3 ids matched to more than one S1 | **0** | One S1 plus its matches is a closed **cluster**, a clean unit to split on |
| S1 entities with no match (singletons) | **5.59%** | They must be sampled too, because they score 1.0 or 0.0 on the leaderboard |
| S2/S3 records that match no S1 (distractors) | **26.0%** (1,340,997 in S2, 1,340,857 in S3) | They must be sampled too, or blocking faces no wrong candidates |
| Matches per S1 | 3.46 on average (0–10) | |
| Country of S1 records | 60.0% US, 40.0% India | France appears only in test |

## How it works

1. **Bucket** every id with `crc32(id) % 100`. This is deterministic: the same id always lands in the same bucket, on any machine, with no stored state.
2. **Clusters follow their S1.** An S1 record, its ground-truth row and all its matched S2/S3 records go to the S1 id's bucket.
3. **Distractors use their own id's bucket.** A tier with k% of the buckets therefore gets about k% of the clusters *and* k% of the distractors, which keeps the same mix as the full data.
4. **Tiers are sets of buckets**, written in one streaming pass over each file (about 22 seconds for all tiers, 800 MB peak memory).

Each tier is written to `data/splits/<tier>/` as `source1.tsv`, `source2.tsv`, `source3.tsv` and `ground_truth.tsv`, in the same format as the original files. `data/` is git-ignored.

## Tiers

Measured after `--make-splits`:

| Tier | Buckets | S1 | S2+S3 | Singletons | Distractors | US / India | Use |
|---|---|---|---|---|---|---|---|
| `val_1` | 0 | 22,182 | 103,713 | 5.55% | 26.02% | 60.3 / 39.7 | Smoke tests and fast iteration: the pipeline runs in about 3 s |
| `val_5` | 0–4 | 110,369 | 516,164 | 5.47% | 25.94% | 60.2 / 39.8 | Compare models, tune thresholds |
| `val_10` | 0–9 | 220,730 | 1,032,115 | 5.51% | 25.98% | 60.0 / 40.0 | The final local number; run rarely |
| `train_10` | 10–19 | 220,367 | 1,029,967 | 5.66% | 26.00% | 60.1 / 39.9 | Fast training while iterating |
| `train` | 10–99 | 1,986,091 | 9,288,104 | 5.59% | 25.99% | 60.0 / 40.0 | Full training |
| *full data* | | 2,206,821 | 10,320,219 | 5.59% | 26.0% | 60.0 / 40.0 | |

Every tier has 3.46 matches per S1, the same as the full data.

## Guarantees

Checked by `tests/test_split.py`:
- **Deterministic:** buckets never change between runs.
- **Nested:** `val_1 ⊂ val_5 ⊂ val_10` and `train_10 ⊂ train`, so a smaller tier's score is a noisy preview of the larger one.
- **No leakage:** `val_10` and `train` are disjoint and together cover everything.
- **Self-contained:** every true match of every S1 in a tier is in that tier's S2/S3 files, and every ground-truth row belongs to an S1 in the tier.
- **Representative:** on the real data, `val_5` is within 1 point of the full data for singletons and distractors.

## Usage

```bash
uv run python main.py --make-splits            # write every tier to data/splits/ (~22 s)
uv run python main.py --subset val_1           # run the pipeline on a tier
uv run python main.py --subset val_5 --nrows 1000
```

In code:

```python
from er.evaluate import F05Evaluator
truth = F05Evaluator.load("data/splits/val_1/ground_truth.tsv")
```

## Rules of thumb

- **Never fit anything on buckets 0–9.** That includes word tables mined from the ground truth and thresholds you "just check" on validation. Tune on `val_5`, then confirm once on `val_10`.
- **Try out on `val_1`, decide on `val_5`/`val_10`.** A 1% tier has 1% of the businesses per city, so blocking faces far fewer look-alike records than on the full data. Blocking precision and recall on `val_1` will look optimistic. Report blocking recall on more than one tier to see the trend as density grows.
- **France can't be validated locally**, because it isn't in train. Only the leaderboard measures it. A leave-one-country-out check (fit on US, evaluate on India) would approximate how well rules transfer to an unseen country; it isn't built yet.

## Changing the tiers

Add an entry to `HashSplitter.tier_buckets` (a name mapped to a `range` of buckets), then re-run `--make-splits`. Keep new validation tiers inside buckets 0–9 and new training tiers inside 10–99. The `--subset` flag picks up new names automatically.
