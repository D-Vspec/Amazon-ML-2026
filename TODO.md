# TODO — Cleaning Stage

Status: [ ] todo · [~] in progress · [x] done

## Setup
1. [x] git init + initial commit
2. [x] Project structure: data/, src/, tests/, output/, notebooks/
3. [x] Sample train TSVs, catalogue noise patterns (see CLAUDE.md)
4. [x] Write CLAUDE.md
5. [x] Write TODO.md
6. [ ] **Get approval on open decisions below before writing clean.py**
7. [ ] Python env: pick interpreter (3.11 available), create venv, pin deps in `requirements.txt`
       (pytest + transliteration lib)

## Base normalization (shared by name + address)
8.  [ ] `normalize_null(s)` — None/NaN/`<NULL>`/whitespace → `""`
9.  [ ] `base_normalize(s)` — NFKC → transliterate → strip diacritics → lowercase → collapse spaces
10. [ ] `transliterate(s)` — all Indic scripts seen (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati) → Latin
11. [ ] `strip_accents(s)` — é→e, è→e, ç→c, Ó→O, etc.
12. [ ] Tests for 8–11 (incl. determinism: repeated calls identical; empty input)

## Name cleaning
13. [ ] Punctuation rules: `&→and`, `. , - / ( ) [ ] # --` → space; drop `m/s` prefix
14. [ ] Legal-suffix table: corp, pvt, ltd, inc, co, llc/l.l.c., llp, pllc, pc/p.c., lp, opc,
        + transliterated Indic forms (`limited`, `li.`, `praivet`, …) — tune from real transliteration output
15. [ ] URL-as-name handling: `foo.com` → `foo` (strip scheme/www/TLD)
16. [ ] Alias markers (`dba`, `d/b/a`, `fka`, `formerly`, `aka`) — see decision D3
17. [ ] Collapse immediately repeated tokens (`exim exim` → `exim`)
18. [ ] `clean_name(s)` tests using real rows from each source

## Address cleaning
19. [ ] Junk removal: `##`, `<null>`, `fl 0` / `fl. 0`, leading zeros on house numbers
20. [ ] Abbreviation table: rd, st*, ave, blk, dr, ln, ct, pl, blvd, hwy, apt, flr, bldg, no, h no,
        nagar/marg variants, etc. (*St: Street vs Saint — context rule)
21. [ ] Landmark phrases: normalize `nr`/`opp.`/`near` → canonical, keep following phrase intact;
        don't treat a bare `opp` city as a landmark
22. [ ] Component extraction → `{postal, city, state, country_token, rest}` — see decisions D1, D2
23. [ ] `clean_address(s, country)` tests using real rows (incl. reordered components)

## Country cleaning
24. [ ] `clean_country(s)` — null handling, NFKC, lowercase, strip. Free text, no fixed set.
25. [ ] Tests incl. `France`, unseen values, empty

## Integration
26. [ ] `clean_record(dict) -> dict` wrapper (never drops fields/rows)
27. [ ] Notebook/script: run cleaners on 1k sampled rows per source, eyeball before/after
28. [ ] Throughput check on a full source file (must be tractable for ~22M rows)
29. [ ] Final review of CLAUDE.md + TODO.md, commit

## Open decisions (need your call)
- D1 State normalization: hand-written local alias tables (e.g. `il`↔`illinois`, `gj`↔`gujarat`)?
      They are local data, not external lookup — but they are country-specific. Unknown
      countries (France) would fall back to no state extraction.
- D2 Postal codes: data has almost none. Extract ONLY clearly-shaped cases (Indian 6-digit at
      end of address / after "pin"; US `ddddd-dddd`), else leave empty? (Recommended.)
- D3 Aliases (`X d/b/a Y`): keep full string, or also split into primary/alias fields?
- D4 Legal suffixes: expand (`llc` → `limited liability company`) or canonicalize to one token
      (`l.l.c.` → `llc`)? Brief says expand for corp/pvt/ltd/inc/co; unclear for llc/llp/pc.
- D5 Transliteration library: `anyascii` (ISC, offline, all scripts, deterministic) vs
      `indic-transliteration` (needs script detection per string). Recommend anyascii.
