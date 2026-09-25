# TODO — Cleaning Stage

Status: [ ] todo · [~] in progress · [x] done

## Setup
1. [x] git init + initial commit
2. [x] Project structure: data/, src/, tests/, output/, notebooks/
3. [x] Sample train TSVs, catalogue noise patterns (see CLAUDE.md)
4. [x] Write CLAUDE.md
5. [x] Write TODO.md
6. [x] Decisions approved (see CLAUDE.md "Decisions")
7. [x] Python 3.11 venv, `requirements.txt` (anyascii, pytest, hypothesis)

## Base normalization
8.  [x] Null handling (None / NaN / `<NULL>` / `nan` → "")
9.  [x] NFKC → anyascii (all Indic scripts + accents) → lowercase, inside every public function
10. [x] Tests incl. determinism and empty input

## Name cleaning
11. [x] Punctuation, `&`/`+` → and, `M/s` prefix dropped
12. [x] Legal forms kept short (+ anyascii Indic spellings, Hindi `प्रा. लि.`)
13. [x] Website names (`foo.com`, `www.`), dotted initials (`l.l.c.`)
14. [x] Alias markers: keep part after marker, case-sensitive exact list (verified on ground truth)
15. [x] Collapse back-to-back repeated words

## Address cleaning
16. [x] Junk: `##`, `<null>`, `Fl 0`, leading zeros, `N°`
17. [x] Abbreviations (rd, ave, blk, dr*, st*, ...) with St/Dr/Ste context rules
18. [x] `opp` kept short; landmark phrases intact
19. [x] West/East suffix rewrite before comma split; orphan fragments
20. [x] Components: postal_code (unmistakable only), state (last part / next to country), city, country
21. [x] Old→new Indian city names, whole part only

## Country
22. [x] `clean_country`: open set, normalized once in `clean_record`; test that no lookup uses raw country

## France
23. [x] France tables restored verbatim from the old project (re-keyed to lowercase "france").
        Teammate-owned — do not modify.

## Integration
24. [x] `clean_record(dict) -> dict` (never drops fields)
25. [x] Property tests: idempotence, ASCII output, never raises
26. [x] `notebooks/sample_clean.py` → `output/clean_samples.md` (20/country/source + coverage)
27. [x] `notebooks/spot_check.py` — seeded random spot check
28. [x] Postal codes: accepted at 0% — source data contains none (noted in CLAUDE.md)

## CLEANING STAGE: COMPLETE except item 29

29. [ ] **State position rule (3b)** — teammate confirming approach. Current rule (last part /
        next to country) misses 13.2% of US/India rows: state is the FIRST part (5.9%) or a
        MIDDLE part (7.3%). Every non-empty address has a state part somewhere. Behaviour unchanged.

## Noted, not actioned (for later)
- `4 Ter Rue …` (French "ter" = house-number suffix) → `4 terrace rue …` via the general `ter` → terrace.
- City = nearest digit-free part before the state; weak when the part is a street
  (`Lawerence Road, Delhi` → city `lawerence road`).
- Throughput 7.8k–17.8k records/s single process (varies with machine load) → ~20–47 min for
  ~22M rows. Parallelize when running on full data.

## Next (not started — awaiting go-ahead)
- Blocking stage.
