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
│   └── normalize.py        # RuleNormalizer: name_norm, legal_form, address_norm
├── tests/                  # pytest tests, one file per module
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

## Running

```bash
uv run python main.py --nrows 1000                 # quick run on the first 1000 rows of each source
uv run python main.py --split test                 # full test split
uv run python main.py --transliterator anyascii --normalizer rules   # pick implementations by name
```

## Adding an implementation

1. Write a class in `src/er/` with the same methods as the existing one for that stage (e.g. `transform(df) -> df` for a transliterator).
2. Register it in the matching dict in `main.py` (e.g. `TRANSLITERATORS`).
3. Select it with the CLI flag. No other file changes.
