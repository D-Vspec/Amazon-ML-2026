# Blocking

For each S1 record, find a short list of likely S2/S3 matches (candidates), so the matcher scores 20 pairs per S1 instead of millions.

Code: `src/er/blockers/tfidf.py` (`TfidfNgramBlocker`). Tests: `tests/test_blocker_tfidf.py`. Scoring: `F05Evaluator.blocking_recall` ([evaluation.md](evaluation.md)).

## Why

- **Comparing everything is impossible.** The test set has 1.7M S1 and 10M S2/S3 records, about 10¹³ pairs.
- **Blocking caps recall.** Blocking keeps only the most similar records. A true match it drops can never be predicted by the matcher, so **recall@k** (the share of true pairs among the top k candidates) is the number that matters here.
- **Precision isn't blocking's job.** Wrong candidates are fine at this stage; the matcher removes them.

## How the TF-IDF blocker works

1. **Text:** `name_norm + " " + address_norm`, taken from the [normalizer](normalization.md).
2. **Vectors:** each record becomes a TF-IDF vector over its **character trigrams** (`char_wb`, so trigrams don't cross word boundaries). Trigrams tolerate typos (`federation` / `fdeeant` still share several), reordering and partial words.
   - `sublinear_tf`: a trigram repeated in one record counts log-scaled.
   - `max_df=0.05`: trigrams present in more than 5% of records (`" st"`, `"roa"`, `"ltd"`) are dropped. They match almost everything and cost the most compute.
   - The vocabulary is fitted on the S2+S3 records.
3. **Same country only.** The S2+S3 vectors are split by the `country` string, and each S1 is compared only with its own country. `country` is an open set: any label works, including France, and an S1 whose country has no S2/S3 records gets no candidates.
4. **Top-k by cosine similarity.**
   - The target vectors sit in a sparse matrix on the chosen device.
   - S1 vectors are sent in batches sized so that at most 250M scores exist at once (~1 GB).
   - Each batch is one sparse × dense multiply, followed by `torch.topk`.
5. **Output:** `s1_id, cand_id, score`, best first, at most `BLOCK_K` rows per S1. Pairs with zero similarity are dropped.

### Why these choices

Measured on a sample of 5,000 S1 records from `set_0`:

| Text | Trigrams | max_df | Recall@5 | Recall@20 | Search |
|---|---|---|---|---|---|
| name only | 3 | 1.0 | 0.630 | 0.761 | 8 s |
| **name + address** | 3 | 1.0 | 0.913 | 0.976 | 31 s |
| name + address | 2–4 | 1.0 | 0.912 | 0.977 | 113 s |
| name + legal form + address | 3 | 1.0 | 0.915 | 0.978 | 32 s |
| name + address | 3 | 0.2 | 0.913 | 0.977 | 29 s |
| **name + address** | **3** | **0.05** | **0.911** | **0.977** | **15 s** |
| name + address | 3 | 0.02 | 0.900 | 0.966 | 5 s |

- **Address:** adding it lifts recall@20 from 76% to 98%. Many true matches have a mangled or replaced name but the same address.
- **Trigram range:** 2–4-grams gain nothing and are about 4× slower.
- **Legal form:** it's within noise, so it's left out.
- **`max_df=0.05`:** it halves the search time with no recall loss. At 0.02, recall starts to drop.

## Results on `set_0`

The full set: 220,907 S1 records, 1,032,493 S2+S3 records.

| | Value |
|---|---|
| recall@1 | 0.269 |
| recall@5 | 0.910 |
| recall@10 | 0.968 |
| **recall@20** | **0.977** |
| Candidate pairs | 4,418,140 (20 per S1) |
| Blocking time, GPU (RTX 5060 Laptop, 8 GB) | 220 s in total: 17 s fitting, 204 s search (1,086 queries/s), 2.5 GB GPU memory |
| Search time, CPU (16 threads, `sparse_dot_topn`, for comparison) | 925 s (239 queries/s) |

Recall@1 can't go much above 0.29, because an S1 has 3.46 true matches on average and only one can be ranked first.

## Density: one set vs three sets

More data means more look-alike records competing for the top 20. Measured with `SETS=0` and `SETS=0,1,2`:

| | `SETS=0` | `SETS=0,1,2` |
|---|---|---|
| S1 / S2+S3 records | 220,907 / 1,032,493 | 662,028 / 3,096,702 |
| recall@5 | 0.910 | 0.891 |
| recall@10 | 0.968 | 0.957 |
| recall@20 | 0.977 | **0.968** |
| Blocking time (GPU) | 220 s | **1,992 s** |
| Peak RAM (whole pipeline) | 2.6 GB | 5.9 GB |

- **Recall** drops about 1 point per 3× data. Expect the full data to lose a few more points, so tune `BLOCK_K` on more than one set.
- **Time** grows with S1 × S2/S3 records: 3× the data took 9× as long.

## Device

`DEVICE=cuda` in `.env` runs the similarity search on the GPU, and falls back to the CPU when no GPU is available. The code is identical on both. A test checks that CPU and GPU return the same candidates and scores. On a 5,000-query sample the CPU does 277 queries/s against the GPU's 1,086/s. Half precision (fp16) was 22% faster but risks reordering near-ties, so the search stays in fp32.

## What it misses

On a 5,000-query sample, 402 of 17,165 true pairs (2.3%) aren't in the top 20:
- **77% are Indian records.** Most are Hindi transliterations of English words, so the name shares almost no trigrams, and often the address is also shorter:
  - `perfect media` ↔ `prphekt midiya`
  - `real consultancy` ↔ `riyl kmsltemsi`
  - `black developers` ↔ `blaik devlprs`
  - `my enterprises` ↔ `may emtrpraijej`
- **161 have an empty candidate address**, leaving only a name to compare.
- **172 share no name word at all.** These include US records whose name was replaced by an unrelated one (`arabele dejesus biotherapeutics` ↔ `keloarcdelta`, same address).

## Limitations and next steps

- **Transliterated English words** need either a word table mined from the ground truth (for the normalizer), or a second blocker that understands them, such as multilingual embeddings. With blockers combined as a union, the misses of one can be covered by another.
- **Full test set:** it has 1,732,544 S1 and 9,969,589 S2/S3 records, 7.8× and 9.7× one set, so about 76× the work of one set given the measured scaling. The run hasn't been measured yet. Speeding it up (for example grouping by city as well as country, or an approximate nearest-neighbour index) is the next blocking task.
- **Memory:** three sets peaked at 5.9 GB of RAM, close to this machine's free memory. The full data needs streaming or a larger machine.

## Adding another blocker

1. Create `src/er/blockers/<name>.py` with a class that has `fit(targets)` and `query(s1_records, k)`, returning `s1_id, cand_id, score` (the contract in `CLAUDE.md`).
2. Register it in `BLOCKERS` in `main.py`, then set `BLOCKER=<name>` in `.env`.
3. Run on the same sets, compare the recall@k that `main.py` prints, and add the numbers here.
