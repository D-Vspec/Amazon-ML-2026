# Evaluation

How to score the pipeline locally the same way the leaderboard does.

Code: `src/er/evaluate.py` (`F05Evaluator`). Tests: `tests/test_evaluate.py`. Data to score on: [data_splits.md](data_splits.md).

## The metric

The leaderboard uses **macro F0.5**: a score for each Source 1 entity, averaged over **all** S1 entities.

```
P = |predicted ∩ true| / |predicted|
R = |predicted ∩ true| / |true|
F0.5 = 1.25 · P · R / (0.25 · P + R)
```

- **Singletons** (no true matches) score **1.0** if you predict nothing and **0.0** if you predict anything.
- An entity with true matches scores 0.0 if nothing predicted is correct, including an empty prediction.
- β = 0.5 counts precision twice as much as recall.

The README's worked example: predict `S2-00047, S2-00193, S3-00812` when the truth is `S2-00047, S3-00812`. That's P = 2/3 and R = 1, so F0.5 = **0.7143**. This is a unit test.

## How much precision matters

Measured on `set_0` (220,907 S1 entities) by corrupting the ground truth:

| Prediction | Macro F0.5 |
|---|---|
| Exactly the ground truth | **1.0000** |
| Ground truth plus **one wrong id** for every entity | **0.7515** |
| Only **one** true match per entity (blocking recall 27%) | **0.6965** |
| Nothing at all | **0.0558** (= the singleton rate) |

One wrong id per entity costs about 25 points. Keeping only a quarter of the true matches, all correct, costs about 30. When the matcher is unsure, it should leave the id out.

## API

```python
from er.evaluate import F05Evaluator
e = F05Evaluator()

truth = e.load("data/splits/set_0/ground_truth.tsv")   # {s1_id: set(ids)}
pred = e.load("output/matching_results.tsv")           # same format as the submission file
e.score(pred, truth)                                   # macro F0.5

cands = e.load("output/candidate_pairs.tsv")
e.blocking_recall(cands, truth)   # share of true (S1, match) pairs in the candidates
e.mean_candidates(cands, truth)   # average candidate-list length, i.e. the matcher's workload
```

`load` reads any `source1_entity_id <TAB> comma-separated ids` file: the ground truth, `matching_results.tsv` or `candidate_pairs.tsv`. Empty lists become empty sets, and duplicate ids are collapsed.

## Behaviour details

Tested in `tests/test_evaluate.py`:
- **Scored entities:** the average is over every S1 in the **truth**. An S1 missing from the prediction counts as an empty prediction. The real scorer rejects a submission that is missing S1 rows; the validator in `utils/validate_submission.py` catches that.
- **Extra entities:** S1 ids in the prediction that aren't in the truth are ignored. This lets you score a full-test-format prediction against a validation tier.
- **Blocking recall** is pooled over all true pairs, not averaged per entity. It's the ceiling on matcher recall: a true match missing from the candidates can never be predicted. Singletons have no true pairs, so they don't affect it.

## Workflow

1. Run `uv run python main.py --make-splits` once.
2. Develop and score on one set (`--sets 0`). Use more sets when two versions are close.
3. Fit anything learned on sets you don't score on.
4. Report blocking recall and mean candidates alongside F0.5, since together they show where the points are lost.
