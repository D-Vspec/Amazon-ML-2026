# CLAUDE.md — Business Entity Resolution: Cleaning Stage

Challenge spec: `6ab10eb3b23ba_student_resource/student_resource/README.md`.
Raw data: `6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv`
(tab-separated, `sep="\t"`, 200–500 MB per file — stream or sample, don't load blindly).

`Amazon-ML-2026-main/` is a separate earlier project. Do not import from or edit it.

## Current scope: text cleaning ONLY

Build `src/clean.py`: pure functions that turn raw `business_name`,
`business_address`, `country` strings into cleaned strings (plus extracted
address components). No blocking, features, similarity, or models in this stage.

## Hard rules

- No external data lookup: no geocoding, APIs, web search. Libraries must work offline.
- Deterministic: same input → same output. No randomness, no dependence on dict/set ordering.
- Pure functions: no I/O, no globals mutated, no side effects.
- Never drop rows. Cleaning transforms fields; it never filters records.
  Empty / `<NULL>` input → `""`, never an exception.
- `country` is an open set. Test has `France` (not in train). Never hard-code `{US, India}`.
- Every non-trivial function has pytest tests, using real examples from the data.

## Observed noise (from train samples — drives the rules)

Names:
- Scripts: Latin + Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati (e.g. `लिमिटेड`, `లిమిటెడ్`).
- Spurious accents on any country's names: `Ówl`, `ÍNC`, `Sólar`, `Léarning`.
- Legal suffixes: moved to the front (`PLLC Novent Owl`), dotted (`l.l.c.`, `inc.`, `p.c.`), abbreviated (`Pvt`, `Ltd`, `Co`, `Corp`, `लि.`).
- Prefixes/junk: `M/s`, `--`, `#`, bracketed fragments `[Owl]`, `(Edm)`.
- URLs as names: `sjacevendome.com`, `AVISSOLUTIONS.COM`.
- Aliases: `d/b/a`, `fka`, `formerly`, `aka`.
- Repeated tokens: `Haven Exim Exim`. Typos (`Seraices`) — NOT fixed here (later fuzzy features handle them).

Addresses:
- Case chaos, component reordering (`Des Plaines, IL, 9308 Home Court`).
- Junk: `##404`, `<NULL>`, `Fl 0` / `Fl. 0`, zero-padded house numbers (`003808`).
- `St` = Street OR Saint (`ST LOUIS`, `St Charles`) — must be context-dependent.
- States as code / full name / native script: `IL`/`Illinois`, `GJ`/`Gujarat`/`ગુજરાત`.
- Landmarks: `Near X`, `Nr X`, `Opp X`, `Opp.X` — keep intact. But `Opp, AL` is a city.
- Essentially NO postal codes in train addresses: 5–6 digit numbers are house numbers.
  Postal extraction must not guess from bare digits.

## Pipeline order (per field)

Each public function runs the whole chain itself — no ordering dependency between functions.

1. Null handling (`None`, NaN, `<NULL>`, `nan`, whitespace-only → `""`).
2. Unicode NFKC.
3. `anyascii` transliteration: all Indic scripts → Latin AND accents stripped (`é→e`, `ç→c`).
4. Lowercase (names: alias markers are resolved BEFORE this — they are case-sensitive).
5. Punctuation: `&`/`+` → `and`; everything outside `[a-z0-9]` → space
   (addresses keep `/` and `-` only between digits: `26/34`, `54-18-45`).
6. Token tables (whole-token only), then collapse whitespace.

## Decisions (confirmed)

- Legal forms kept SHORT: private→pvt, limited→ltd, corporation→corp, incorporated→inc, company→co.
- Alias markers (`fka`, `f/k/a`, `formerly`, `aka`, `dba`, ... — exact case-sensitive list in code):
  keep the part AFTER the marker (ground truth: >99.9% of 97k cases). `Aka`/`AKA`/`Dba` are
  name words, not markers. Marker at the start → no split.
- `M/s` prefix dropped; back-to-back repeated name words collapsed.
- `opp`/`opp.`/`opposite` → `opp`, never expanded (Opp, AL is a city).
- `(we, st)`/`(w)`/`(west)` → `west`, `(ea, st)`/`(e)`/`(east)` → `east`, on the raw string
  before comma splitting; orphan `st)` parts dropped, dangling `(ea`/`(we` → east/west.
- `Fl 0` / `Floor 0` dropped; real floors kept.
- State lookup only on the last part or a part next to the country part (`Washington, DC`).
- Postal codes only when unmistakable (India: 6 digits not starting with 0, at end or after
  "pin"; US: `ddddd` / `ddddd-dddd` at end); otherwise empty. Other countries: never.
- Country normalized ONCE in `clean_record`; downstream functions take the clean lowercase value.
- No France-specific tables (general rules only). If added later, document that they came from
  observing the input distribution, not external data.
- Known limit: name cleaning is idempotent except when the cleaned output contains a lowercase
  alias marker word (`media aka services`) — re-cleaning would split it. Clean raw input once.

## Conventions

- Python 3.11, venv in `.venv/`, deps pinned in `requirements.txt`. Run tests: `.venv/Scripts/python -m pytest`.
- Samples: `.venv/Scripts/python notebooks/sample_clean.py` → `output/clean_samples.md`.
- Abbreviation tables are module-level constants (dicts), sorted, easy to review.
- Tests in `tests/test_clean.py`; each rule has at least one test using a real data example.
- Commit after every meaningful change. Keep `TODO.md` current.
