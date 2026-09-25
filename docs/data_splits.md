# Data splits

The training data is split into **`N_SETS` equal sets**, 10 by default (`set_0` … `set_9`, each about 10% of the data). Run on one set for speed, or on several for a more reliable number.

Code: `src/er/split.py` (`HashSplitter`). Tests: `tests/test_split.py`. Scoring: [evaluation.md](evaluation.md).

## Why sets must keep clusters together

Every S2/S3 record matches **at most one** S1 record (measured on the ground truth: 0 exceptions). An S1 and its matches therefore form a closed cluster. Splitting rows at random would separate an S1 from its matches, so recall would look terrible for no reason.

26.0% of S2/S3 records match nothing. These distractors must be spread over the sets too, otherwise there's nothing wrong to match against and precision looks perfect for no reason.

## The rule

```
set = crc32(id) % N_SETS          # N_SETS = 10 by default
```

- **S1 record, its ground-truth row and all its matches:** the set of the S1 id.
- **Distractor (matches no S1):** the set of its own id.

The rule is deterministic, so the sets are identical on every machine and every run.

## Measured sets

| Set | S1 | S2+S3 | Singletons | Distractors | Matches / S1 | US |
|---|---|---|---|---|---|---|
| `set_0` | 220,907 | 1,032,493 | 5.58% | 26.00% | 3.46 | 60.0% |
| `set_1` | 220,358 | 1,032,494 | 5.57% | 26.01% | 3.47 | 60.0% |
| `set_2` | 220,763 | 1,031,715 | 5.58% | 25.99% | 3.46 | 60.1% |
| `set_3` | 220,122 | 1,029,460 | 5.59% | 26.05% | 3.46 | 60.1% |
| `set_4` | 219,802 | 1,027,078 | 5.57% | 26.03% | 3.46 | 59.8% |
| `set_5` | 220,704 | 1,033,191 | 5.61% | 26.03% | 3.46 | 60.0% |
| `set_6` | 220,348 | 1,029,877 | 5.57% | 26.01% | 3.46 | 59.9% |
| `set_7` | 220,609 | 1,032,113 | 5.58% | 25.93% | 3.47 | 59.9% |
| `set_8` | 222,089 | 1,037,189 | 5.65% | 25.89% | 3.46 | 60.1% |
| `set_9` | 221,119 | 1,034,609 | 5.57% | 25.91% | 3.47 | 59.8% |
| full data | 2,206,821 | 10,320,219 | 5.59% | 26.0% | 3.46 | 60.0% |

The rest of each row is India. France appears only in the test data.

## Usage

Pick sets in `.env`, then run `uv run python main.py`:

```bash
SETS=0          # one set: pipeline in ~27 s
SETS=0,1,2      # three sets, concatenated: ~79 s
```

The first run writes `data/splits/10_sets/set_0..set_9` automatically (~20 s, git-ignored).

`N_SETS` in `.env` changes the number of sets. Each count gets its own folder (`data/splits/<N>_sets/`), because a different count assigns records to different sets. `SETS` must stay below `N_SETS`. The measured numbers above are for the default of 10.

Each `data/splits/<N>_sets/set_k/` holds `source1.tsv`, `source2.tsv`, `source3.tsv` and `ground_truth.tsv`, in the same format as the original files. Score with `F05Evaluator.load("data/splits/10_sets/set_0/ground_truth.tsv")`.

**Fit on sets you don't score on.** Anything learned from data (a model, a mined word table, a threshold) must be scored on a different set than it was fitted on.

## Guarantees

Checked by `tests/test_split.py`:
- **Deterministic:** the rule never changes between runs, and the sets are roughly equal in size.
- **No overlap:** every record is in exactly one set.
- **Self-contained:** every true match of every S1 in a set is in that set.
- **Representative:** on the real data, sets are within 1 point of the full data for singletons and distractors.

## Limitations

- A set has 10% of the businesses per city, so blocking faces fewer look-alike records than on the full data. Its candidate recall will look slightly optimistic, and running on more sets shows the trend.
- France can't be validated locally. Only the leaderboard measures it.
