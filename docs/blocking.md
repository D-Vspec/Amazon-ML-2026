# Blocking

For each S1 record, find a short list of likely S2/S3 matches (candidates), so the matcher scores a few dozen pairs per S1 instead of millions.

There are two blockers, and by default they run together as a union (`BLOCKER=tfidf+embedding`):
- **TF-IDF over character trigrams** of the normalized text. Good at typos, numbers and exact strings.
- **A multilingual embedding model** on the raw, untransliterated text. Good at native scripts, where transliteration destroys the signal.

Code: `src/er/blockers/tfidf.py` (`TfidfNgramBlocker`), `src/er/blockers/embedding.py` (`EmbeddingBlocker`), `src/er/blockers/union.py` (`UnionBlocker`). Tests: `tests/test_blocker_*.py`. Scoring: `F05Evaluator.blocking_recall` ([evaluation.md](evaluation.md)).

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

## What TF-IDF misses

On a 5,000-query sample, 402 of 17,165 true pairs (2.3%) aren't in the top 20:
- **77% are Indian records.** Most are Hindi transliterations of English words, so the name shares almost no trigrams, and often the address is also shorter:
  - `perfect media` ↔ `prphekt midiya`
  - `real consultancy` ↔ `riyl kmsltemsi`
  - `black developers` ↔ `blaik devlprs`
  - `my enterprises` ↔ `may emtrpraijej`
- **161 have an empty candidate address**, leaving only a name to compare.
- **172 share no name word at all.** These include US records whose name was replaced by an unrelated one (`arabele dejesus biotherapeutics` ↔ `keloarcdelta`, same address).

## The embedding blocker

**Why raw text.** The transliterator turns `परफेक्ट मीडिया` into `prphekt midiya`, which shares almost nothing with `perfect media`. A multilingual model reads the Devanagari directly and places it next to "Perfect Media": cosine 0.94 in `tests/test_blocker_embedding.py`, where a Tamil `லக்ஷ்மி சர்வீசஸ்` also finds `Lakshmi Services`. So this blocker skips the transliterator and normalizer. `main.py` keeps the original text as `raw_name` and `raw_address` for it.

**How it works:**
1. **Text:** `"query: " + business_name + " | " + business_address`, as it appears in the source files. The `query: ` prefix is what E5 models expect.
2. **Model:** `intfloat/multilingual-e5-small` (MIT, 118M parameters), set by `EMBED_MODEL` in `.env`. It runs in fp16 on the GPU and fp32 on the CPU, and vectors are L2-normalized.
3. **Same country only**, top-k by cosine similarity. It uses a dense matrix multiply plus `torch.topk`, batched like the TF-IDF blocker.

**Multilingual on raw text vs English on transliterated text.** Measured on the full `set_0` (220,907 queries), recall@20:

| Model and input | All | India | US | Time (GPU) |
|---|---|---|---|---|
| **`multilingual-e5-small`, raw text** | **0.978** | **0.963** | **0.988** | 148 s |
| `bge-small-en-v1.5` (MIT, 33M), transliterated + normalized text | 0.960 | 0.929 | 0.981 | 171 s |

The English model misses almost twice as many Indian pairs (7.1% vs 3.7%), because it only ever sees the mangled transliteration. The multilingual model is better in the US too, and faster in this run.

## Union of both

`UnionBlocker` asks each member for its own top `BLOCK_K`, merges the lists and removes duplicate pairs. Each pair keeps one score per member (`score_tfidf` and `score_embedding`, 0 when that member didn't propose it) plus `score` = the higher of the two. The matcher can therefore learn how far to trust each signal.

Measured end to end with `uv run python main.py` on the full `set_0`:

| Blocker | Recall | Candidates per S1 | Blocking time (GPU) |
|---|---|---|---|
| `tfidf` | 0.977 | 20 | 220 s |
| `embedding` | 0.978 | 20 | ~150 s |
| **`tfidf+embedding`** | **0.990** | **34.6** | **375 s** |

The union cuts missed true pairs from 2.3% to 1.0%. The two lists overlap heavily, giving 34.6 candidates per S1 rather than 40. Peak RAM for the whole run was 4.5 GB.

### Who finds what

True pairs on the full `set_0`, by which blocker put them in its top 20:

| | Both | Embedding only | TF-IDF only | Neither |
|---|---|---|---|---|
| India | 93.83% | **2.48%** | 1.74% | 1.95% |
| US | 98.28% | 0.49% | 0.82% | 0.40% |

**Only the embedder finds** native-script records and heavy typos:
- `Seven Sun Producer Private Limited` ↔ `सेवन सन प्रोड्यूसर प्राइवेट लिमिटेड`
- `International Construction Private Limited` ↔ `इंटरनेशनल कंस्ट्रक्शन प्राइवेट लिमिटेड`
- `Urban Exports Pvt Ltd` ↔ `அர்பன் Exports Pvt Ltd`
- `Seven Sun Producer Private Limited` ↔ `Priavfe Seven Sun Producer Limited`

**Only TF-IDF finds** (in the examples inspected) mostly near-exact names whose candidate has an empty address:
- `Patna Properties Private Limited` ↔ `Patna Properties Prevate Limited` (no address)
- `Target Trading Private Limited` ↔ `TARGET TRADING PRIVATE PRIVATE LTD` (no address)
- `Good Products Private Limited` ↔ `ಗುಡ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` (one Kannada-script case the embedder missed)

**Neither finds:**
- Generic names in native script in dense cities, where many look-alikes crowd the top 20: `Global Infotech Limited` ↔ `ग्लोबल इंफोटेक लिमिटेड`, `Modern Infotech Private Limited` ↔ `ಮಾಡರ್ನ್ ಇನ್‌ಫೋಟೆಕ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- Names replaced by invented words: `Global Traders Private Limited` ↔ `Mirahal0`, `American Legacy Islands Inc` ↔ `Veragildsyn`, with the same street address

## Limitations and next steps

- **The remaining 1% of misses** are mostly generic names crowded out of the top 20 and replaced names. Options:
  - a larger `BLOCK_K` for the embedder
  - an address-only blocker (house number + street) for records whose name was replaced
  - fine-tuning the embedder on ground-truth pairs from sets it isn't evaluated on
- **Time:** the union costs 375 s per set, against 220 s for TF-IDF alone.
- **Full test set:** it has 1,732,544 S1 and 9,969,589 S2/S3 records, 7.8× and 9.7× one set, so about 76× the work of one set given the measured scaling. The run hasn't been measured yet. Speeding it up (for example grouping by city as well as country, or an approximate nearest-neighbour index) is the next blocking task.
- **Memory:** three sets peaked at 5.9 GB of RAM, close to this machine's free memory. The full data needs streaming or a larger machine.

## Adding another blocker

1. Create `src/er/blockers/<name>.py` with a class that has `fit(targets)` and `query(s1_records, k)`, returning `s1_id, cand_id, score` (the contract in `CLAUDE.md`).
2. Register it in `BLOCKERS` in `main.py`, then set `BLOCKER=<name>` in `.env`, or add it to the union: `BLOCKER=tfidf+embedding+<name>`.
3. Run on the same sets. `main.py` prints recall for the top-k, for all candidates, and "found by" for each member of a union. Add the numbers here.
