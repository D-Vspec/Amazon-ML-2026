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

## Integration
23. [x] `clean_record(dict) -> dict` (never drops fields)
24. [x] Property tests: idempotence, ASCII output, never raises
25. [x] `notebooks/sample_clean.py` → `output/clean_samples.md` (20/country/source + coverage)
26. [~] **Awaiting review of samples** before blocking

## Open questions (from sample review)
27. [ ] State position rule (3b) misses 13.2% of US/India rows whose state is the FIRST (5.9%) or a
        MIDDLE (7.3%) part. Every non-empty address has a state part somewhere. Widen the rule?
28. [ ] Postal codes: 0% extracted — the data has no postal codes (train or France test samples).
29. [ ] France: no state/city extracted and `R.`/`All.`/`Rte` not expanded (no France tables, by decision).
30. [ ] Throughput ~7.8k records/s single process → ~47 min for ~22M rows. Parallelize when running on full data.
