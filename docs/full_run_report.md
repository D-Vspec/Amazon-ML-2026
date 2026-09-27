# Full run report: cross-encoder on 80% of train, test submission

The run that produced the leaderboard submission. It ran on a rented vast.ai RTX 4090 on 26–27 Sep 2026, 22:41–11:54 IST. Every number below comes from that run's logs (`logs_vast/`) or from the output files.

## What was run

`full_run.sh` on branch `cross-encoder` (commit `5a5a6d8`):

| Setting | Value |
|---|---|
| Blocker | `tfidf+embedding` (union), `BLOCK_K=20` each ([blocking.md](blocking.md)) |
| Cross-encoder | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M parameters) |
| Fine-tuned on | sets 0, 1, 4, 5, 6, 7, 8, 9 (80% of train) |
| Fine-tuning pairs | `CE_TRAIN_PAIRS=12000000`: every positive, then the hardest negatives by blocker score |
| Matcher | `MATCHER=cross_encoder`: the cross-encoder's probability is the match probability |
| Threshold | tuned on set 2 |
| F0.5 estimate | set 3, which neither training nor tuning saw |
| Saved model | `models/ce_full` (not in git; `model.safetensors` sha256 `7f97bd6a…27d3cf`) |

Stage 1 fine-tunes and evaluates. Stage 2 runs the whole test split with the saved model and the threshold chosen in stage 1. Then `utils/validate_submission.py --check-ids` checks the outputs.

## Machine

vast.ai offer #38381900 (instance 52780549), $0.430/h:

| Part | Spec |
|---|---|
| GPU | RTX 4090, 24 GB |
| CPU | AMD EPYC 7742, 32 vCPU |
| RAM | 129 GB |
| Disk | 80 GB NVMe |
| Software | torch 2.11 + cu128 |

## Held-out result

| Set | Role | F0.5 |
|---|---|---|
| 2 | picks the threshold: **0.40** | 0.9843 |
| **3** | **held-out estimate** | **0.9844** |
| 3 | baseline: predict nothing | 0.0559 |
| 3 | baseline: top blocker candidate only | 0.6301 |
| 3 | previous model (fine-tuned on set 1 only, ~1M pairs, threshold 0.92) | 0.9844 |

- **No gain on set 3.** 12× more fine-tuning data did not move F0.5 (0.9844 both, to 4 decimals). The model is no longer the limit. Blocking recall is (0.990 on set 3: 1% of true pairs are never scored), and so are the remaining hard cases.
- **More confident probabilities.** The new model's best threshold is 0.40 against the old 0.92.
- **Why it was submitted.** It saw 8× more variety of records, which is the safer bet for the unseen France records.
- **Blocking recall on set 3:**
  - top 5: 0.914
  - top 20: 0.978
  - union: **0.990**, with 34.6 candidates per S1
  - TF-IDF alone: 0.977
  - embedding alone: 0.978

## Test submission

| | Value |
|---|---|
| S1 records | 1,732,544 (every one has a row in both files) |
| S2 + S3 records | 4,887,273 + 5,082,316 |
| Candidate pairs scored (`candidate_pairs.tsv`) | 57,278,665, 33.06 per S1 |
| Matches (`matching_results.tsv`) | 5,874,847, 3.39 per S1 |
| S1 with no match | 95,103 (5.49%) |
| Validator (`--check-ids`, on the instance and locally) | **PASS** |
| sha256 `matching_results.tsv` | `4b82eb50fb42f74f51a7fd2b0c566f7e64fea4d8db0d6b8d26bb71b1a5367ec9` |
| sha256 `candidate_pairs.tsv` | `80b7a6bcef474a5bf6322a14b38eb1d9ced6bcbd004d131cc2954a9b115a51b0` |

The test F0.5 is unknown: only the leaderboard has test labels. As a consistency check, set 3 predicted 3.42 matches per S1, against the test's 3.39.

### By country

France is not in the training data.

| Country | S1 | Matches per S1 | No match | Candidates per S1 |
|---|---|---|---|---|
| France | 259,452 | 3.42 | 4.75% | 33.8 |
| India | 809,986 | 3.33 | 5.82% | 33.1 |
| US | 663,106 | 3.45 | 5.38% | 32.8 |

France behaves like the trained countries: similar matches per S1, candidates per S1 and share of unmatched S1. That is a sanity check, not a score.

### Matches per S1

| Matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 records | 95,103 | 117,882 | 307,048 | 415,738 | 367,785 | 241,305 | 120,783 | 47,692 | 14,706 | 3,564 | 938 |

## Timings (RTX 4090)

| Step | Pairs | Time |
|---|---|---|
| Stage 1: block + featurize set 3 | 7,624,275 | 273 s + 252 s |
| Stage 1: block sets 0, 1, 4–9 together | 59,622,978 | 7,692 s (2 h 8 min) |
| Stage 1: fine-tune (1 epoch, batch 64, fp16) | 12M | 10,990 s (3 h 3 min), 1,092 pairs/s, final loss 0.018 |
| Stage 1: score set 3 | 7,624,275 | 1,268 s (6,012 pairs/s) |
| Stage 1: block + featurize + score set 2 | 7,644,070 | 268 s + 279 s + 1,259 s |
| Stage 2: block the test split | 57,278,665 | 7,382 s (2 h 3 min) |
| Stage 2: featurize | 57,278,665 | 2,160 s |
| Stage 2: score | 57,278,665 | 10,686 s (2 h 58 min), 5,360 pairs/s |
| **Whole run** | | **13 h 14 min** (instance about 14 h so far at $0.430/h, about $6) |

## Limitations and what to improve

- **Blocking cost grows with S1 × targets, and blocking caps recall.**
  - Why: TF-IDF search is exact within each country.
  - Measured: the test blocking took 2 h. Changing how many queries go into each GPU call only changes speed by about 25% (laptop, set 3: 174–227 s).
  - Also measured: `sparse_dot_topn` on 16 CPU threads is 4.5× slower than the GPU ([blocking.md](blocking.md)).
  - A real speed-up needs fewer comparisons, for example a finer partition than country, or compressed TF-IDF vectors. Recall then has to be measured.
- **Scoring barely uses the GPU.**
  - Measured: cross-encoder scoring ran at 5,360–6,074 pairs/s with the GPU about 35% busy, only 1.3× the laptop's 4,770 pairs/s.
  - Likely cause: the bottleneck is CPU-side tokenization.
- **The test pool is larger than the evaluation pools.**
  - The test split is blocked as one pool of about 10M targets, while sets 2 and 3 are about 1M each.
  - Earlier measurements lost about 1 point of blocking recall per 3× more data ([blocking.md](blocking.md)), so test recall is probably below set 3's 0.990.
- **Caching only covered training sets.** It has since been added for whole splits (`1a579ff`), so a rerun of the test split with a new model skips blocking and featurizing. This run's test pairs were not cached.

## Reproduce

1. `uv sync`.
2. Run `full_run.sh` (the environment variables at its top are the whole configuration).
3. With `models/ce_full` present, stage 1 skips fine-tuning. Set 2 and set 3 scores come from `CACHE_DIR`.
