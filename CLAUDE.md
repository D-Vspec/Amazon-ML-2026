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

1. Null handling (`None`, NaN, `<NULL>`, whitespace-only → `""`).
2. Unicode NFKC.
3. Transliterate non-Latin scripts to Latin (offline, deterministic library).
4. Strip diacritics (NFKD + drop combining marks): `é→e`, `ç→c`.
5. Lowercase.
6. Punctuation: `&→and`; `. , - / ( ) [ ] #` → space (field-specific exceptions documented in code).
7. Abbreviation expansion via explicit token tables (whole-token match only).
8. Collapse whitespace, strip.

Order matters: transliteration before lowercasing/accent strip; expansion after
punctuation so `Ltd.` and `Ltd` both hit the table.

## Conventions

- Python, stdlib + minimal offline deps (pinned in `requirements.txt`).
- Abbreviation tables are module-level constants (dicts), sorted, easy to review.
- Tests in `tests/test_clean.py`; each rule has at least one test using a real data example.
- Commit after every meaningful change. Keep `TODO.md` current.
