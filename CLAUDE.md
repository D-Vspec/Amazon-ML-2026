# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Business Entity Resolution (ML Challenge 2026)

Challenge spec: `6ab10eb3b23ba_student_resource/student_resource/README.md`. Data is TSV (`sep="\t"`), ~22M records total — check memory before loading full files.

### Modularity (required)

Every pipeline stage must be swappable without touching other stages. Swapping BM25 → TF-IDF → MinHash LSH → embeddings must be a one-line change (a name in the entry point/CLI), never an edit inside another stage.

Pipeline stages and their contracts:

| Stage | Input | Output |
|---|---|---|
| Normalizer | records DataFrame (`entity_id, business_name, business_address, country`) | same columns, cleaned text |
| Blocker | `fit(targets)` on S2+S3 records; `query(s1_records, k)` | pairs DataFrame `s1_id, cand_id, score` |
| Matcher | `fit(pairs, labels)`; `predict(pairs)` | probability per pair |
| Decider | pairs + probabilities, threshold | `{s1_id: [matched ids]}` |

Rules:
- Stages talk only through these DataFrames. No stage reads another stage's internals.
- Blockers are combinable: a union of blockers is itself a blocker (concatenate pairs, dedupe on `s1_id, cand_id`).
- Pick implementations via a plain dict of name → class in the entry point. No plugin frameworks or config systems.
- Each blocker is evaluated on its own with the same function: recall@k on a held-out train split.
- One file per implementation (e.g. `blockers/bm25.py`, `blockers/tfidf.py`, `blockers/lsh.py`).

### Hard constraints (from the challenge)

- No external lookups (geocoding, registries, ER APIs). Local libraries only.
- Final model: MIT/Apache-2.0 license, ≤ 8B params.
- `country` is an open set — test contains `France`, unseen in train. Never hard-code `{US, India}`.
- Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv` (candidates = exactly what the matcher scored). Validate with `utils/validate_submission.py`.
- Metric is macro F0.5 per S1 entity, singletons included — when unsure, predict no match.

### Git

- Never add `Co-Authored-By: Claude` (or any AI attribution) to commit messages or PR descriptions.
