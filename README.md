# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source 1 business record, find the Source 2 / Source 3 records that refer to the same real-world business. Scored by macro F0.5 per Source 1 entity (precision-weighted; singletons count). Full spec: `6ab10eb3b23ba_student_resource/student_resource/README.md` (not tracked — see *Data*).

## Setup

```bash
uv sync              # creates .venv with Python 3.12 and installs the er package (editable)
uv run pytest        # run tests
```

## Data

The challenge data (2.4 GB) is git-ignored. Unzip the student resource into the repo root so this path exists:

```
6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv
```

## Project structure

```
.
├── main.py                 # entry point: instantiates and connects all stages
├── src/er/                 # "entity resolution" package — all pipeline code
│   ├── transliterate.py    # AnyAsciiTransliterator: any script → ASCII
│   ├── normalize.py        # RuleNormalizer: name_norm, legal_form, address_norm
│   ├── split.py            # HashSplitter: 10 equal cluster-aware training sets
│   └── evaluate.py         # F05Evaluator: leaderboard macro F0.5, blocking recall
├── tests/                  # pytest tests, one file per module
├── docs/                   # how each component works and how it was validated (index: docs/README.md)
├── .env                    # pipeline configuration: data, sets, stage implementations
├── data/splits/            # the 10 sets (git-ignored, created automatically)
├── pyproject.toml          # dependencies + package config (uv / hatchling)
├── uv.lock                 # pinned dependency versions
└── CLAUDE.md               # coding guidelines and project rules
```

## Pipeline

Every stage is a class. Stages communicate only through DataFrames, so any implementation can be swapped without touching the others.

| Stage | Status | Input → Output |
|---|---|---|
| Transliterator | ✅ `AnyAsciiTransliterator` | records → records with ASCII name/address |
| Normalizer | ✅ `RuleNormalizer` | records → records + `name_norm`, `legal_form`, `address_norm` |
| Blocker | planned | S1 records + S2/S3 records → candidate pairs `s1_id, cand_id, score` |
| Matcher | planned | candidate pairs → match probability per pair |
| Decider | planned | pairs + probabilities → final matches per S1 |

Records have the columns `entity_id, business_name, business_address, country`.

## Data sets and scoring

The training data is split into 10 equal sets (~220k S1 each) so iterations don't need all 12.5M records:

Set `SETS=0` in `.env` to run on one set, or `SETS=0,1,2` for a more reliable number. The sets are written to `data/splits/` automatically on the first run (~20 s).

Score with `F05Evaluator` from `src/er/evaluate.py`. Details: [docs/data_splits.md](docs/data_splits.md), [docs/evaluation.md](docs/evaluation.md).

## Running

Everything is configured in `.env`; run with:

```bash
uv run python main.py          # or `python main.py` inside the activated .venv
```

| Key | Meaning | Default |
|---|---|---|
| `DATA_DIR` | where the challenge dataset was unzipped | `6ab10eb3b23ba_student_resource/student_resource/dataset` |
| `SETS` | training sets to run on, e.g. `0` or `0,1,2`; empty = the full `SPLIT` | `0` |
| `SPLIT` | `train` or `test`, used when `SETS` is empty | `train` |
| `NROWS` | rows per source file for quick runs; empty = all | empty |
| `TRANSLITERATOR` / `NORMALIZER` | implementation names from the registries in `main.py` | `anyascii` / `rules` |

For a one-off run, an environment variable overrides `.env`: `NROWS=1000 uv run python main.py`.

## Adding an implementation

1. Write a class in `src/er/` with the same methods as the existing one for that stage (e.g. `transform(df) -> df` for a transliterator).
2. Register it in the matching dict in `main.py` (e.g. `TRANSLITERATORS`).
3. Select it by name in `.env`. No other file changes.
