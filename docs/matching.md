# Matching

The matcher decides which blocking candidates are real matches. An XGBoost classifier gives each candidate pair a match probability, and a threshold turns those probabilities into the submission.

Code: `src/er/features.py` (`PairFeaturizer`), `src/er/matcher.py` (`XgbMatcher`), `src/er/decision.py` (`decide`, `tune_threshold`). Tests: `tests/test_features.py`, `tests/test_matcher.py`, `tests/test_decision.py`. Candidates come from [blocking.md](blocking.md), and the scoring is described in [evaluation.md](evaluation.md).

## Pipeline

1. **Features** per candidate pair (`PairFeaturizer`):
   - **From the blocker:** `score` (best member similarity), `rank`, and `score_gap` (distance from the S1's best candidate).
   - **Name:** word Jaccard, rapidfuzz `token_set_ratio` and `partial_ratio`.
   - **Address:** word Jaccard and `token_set_ratio`, house/PIN number Jaccard and overlap, and empty-address flags.
   - **Legal form:** whether the two agree, and whether either is missing.
   - **From the union blocker, passed through:** `score_tfidf` and `score_embedding` (each blocker's real similarity for the pair), `found_by_tfidf` / `found_by_embedding` (which blocker had the pair in its own top 20), and `found_by_all`.
   - **Chunking:** pairs are featurized in chunks of 20,000 S1 records and stored as float32. Doing all 7.6M pairs at once exhausted 22 GB of RAM.
2. **Labels:** a candidate is positive if it's in the ground truth. Every other candidate is a hard negative, because the blocker already rated it similar.
3. **XGBoost** (Apache-2.0): 400 trees, depth 6, positives weighted by the negative/positive ratio. It trains on every feature column it's given and stores their names in the model, so older models keep predicting with their own features.
4. **Decision:** keep pairs with probability ≥ threshold, then give each S2/S3 record only to the S1 that scored it highest. Every S2/S3 record belongs to at most one S1 ([data_splits.md](data_splits.md)).

## Protocol

Three different sets, so the reported score is on data the model and the threshold never saw:

| Role | Set | `.env` |
|---|---|---|
| Train XGBoost | `set_1` | `TRAIN_SETS=1` |
| Choose the threshold (search 0.05–0.85 in 0.05 steps, 0.90–0.99 in 0.01 steps) | `set_2` | `TUNE_SETS=2` |
| Report F0.5 | `set_3` | `SETS=3` |

`set_3` is used for scoring because the teammate's `model.json` may have been trained on `set_0`, the default in `train_matcher.py`.

## Results

All three matchers used the same union blocker (`BLOCKER=tfidf+embedding`, `BLOCK_K=20`), measured on the full sets:

| Matcher | Features | Threshold (from set 2) | F0.5 set 2 | **F0.5 set 3** |
|---|---|---|---|---|
| `model.json` (teammate's, trained on TF-IDF-only candidates) | 14 base | 0.95 (top of the old search range) | 0.9594 | 0.9591 |
| Run A: retrained on union candidates | 14 base | 0.93 | 0.9637 | 0.9632 |
| **Run B: + real per-blocker scores (`model_union.json`)** | 17 | **0.93** | **0.9642** | **0.9638** |
| Baseline: always take the top blocker candidate | | | | 0.6301 |
| Baseline: predict nothing (= singleton rate) | | | | 0.0559 |

- **Retraining on union candidates** gave most of the gain: +0.0041 over `model.json`. The old model had never seen embedding-only candidates, and its `score` feature had changed meaning (it's now the higher of two blockers' scores).
- **Real per-blocker scores** added a further +0.0006, consistently on both set 2 and set 3.
- **Threshold transfer:** in every run, the threshold chosen on set 2 scored within 0.0005 of its set 2 value when applied to set 3.
- **Cost:** each run took about 32 minutes and 8–9 GB of RAM (blocking 3 sets at ~6 minutes each, plus featurizing and ~4 minutes of XGBoost training on the CPU).

### What the models rely on (XGBoost feature importance, top 6)

| `model.json` | | Run A | | Run B | |
|---|---|---|---|---|---|
| score | 0.496 | score | 0.386 | score | 0.433 |
| rank | 0.281 | rank | 0.353 | **found_by_all** | **0.318** |
| addr_token_set | 0.040 | cand_addr_empty | 0.093 | **score_embedding** | **0.114** |
| num_jaccard | 0.029 | addr_token_set | 0.051 | cand_addr_empty | 0.040 |
| name_partial | 0.027 | num_overlap | 0.023 | score_tfidf | 0.017 |
| cand_addr_empty | 0.023 | num_jaccard | 0.018 | rank | 0.016 |

Whether both blockers independently found a pair (`found_by_all`) takes over most of what `rank` used to carry.

## Using the exported model

`model_union.json` at the repo root is run B's model. It must be used with the union blocker, because it expects the `score_tfidf`, `score_embedding` and `found_by_all` columns.
- **Pipeline:** set `MATCHER_MODEL=model_union.json` in `.env`. An existing file is loaded instead of training. A path that doesn't exist yet is trained on `TRAIN_SETS` and saved there.
- **Test set:** `uv run python src/er/predict_test.py --model model_union.json --test-dir <dataset/test> --threshold 0.93`.

## Limitations and next steps

- **Training data:** one set (7.6M pairs, 756,307 true). Training on more sets (`TRAIN_SETS=1,4,5`) is the obvious next experiment. Memory grows with each set.
- **Threshold:** 0.93 was chosen for the training data's density. The test set has 5.75 S2/S3 records per S1 against 4.68 in training ([data_splits.md](data_splits.md)), so check the leaderboard.
- **Training runs on the CPU.** XGBoost supports `device="cuda"`, which isn't enabled yet.
