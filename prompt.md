# Business Entity Resolution — Data Cleaning Stage

## Scope: this session is ONLY text cleaning.

We will do blocking, features, and model training in later sessions.
Do not write any blocking/model code now.

## Hard rules
- NO external data lookup. No geocoding, no APIs, no web search.
- No external libraries that need network access.
- Deterministic only — every cleaning function must produce identical output
  for identical input. No randomness.
- No dropping rows. Cleaning transforms text fields; it never filters records.

## Task: build `src/clean.py`

A module that takes raw business_name, business_address, country strings
and returns cleaned versions. Everything in this module must be:
- Pure functions (input → output, no side effects)
- Unit-tested with pytest
- Reproducible

## Requirements

1. **Name cleaning:**
   - Lowercase
   - Unicode NFKC normalization
   - Strip leading/trailing whitespace
   - Collapse multiple spaces to one
   - Remove or normalize punctuation: & → and, ., ,, -, / → space or ""
   - Expand common legal suffixes:
     - Corp → Corporation
     - Pvt → Private
     - Ltd → Limited
     - Inc → Incorporated
     - Co → Company
     - (and others you find in the data)
   - Handle Hindi transliteration (Devanagari → Latin, deterministic library)

2. **Address cleaning:**
   - Same base normalization as name
   - Expand address abbreviations: Rd → Road, St → Street, Ave → Avenue,
     Blk → Block, etc.
   - Handle French accents (é → e, è → e, ç → c, etc.)
   - Extract and preserve: pincode/postal code, state, city, country tokens
   - Keep landmark phrases intact ("Near SBI ATM") — they're useful signals

3. **Country cleaning:**
   - Lowercase, strip whitespace
   - Do NOT hard-code {US, India}. Test set has France too.
   - Keep as free-text string

## Process

Step 1. `git init` (if not already done). Initial commit.
Step 2. Create the project structure: data/, src/, tests/, output/, notebooks/
Step 3. Read a small sample of each training TSV. Show me 5 rows from each
        with the raw name/address so I can see the noise patterns.
Step 4. Write `CLAUDE.md` describing the cleaning stage. Show me before
        writing code.
Step 5. Write `TODO.md` with a numbered task list for cleaning. Show me.
Step 6. STOP. Wait for my approval before writing clean.py.

## Rules
- Commit to git after every meaningful change.
- Keep TODO.md current.
- pytest for every non-trivial function.
- No subagents.
- After finishing, run /clear before starting the next unrelated task.