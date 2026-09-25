# Normalization

How raw business names and addresses are turned into comparable text before blocking and matching.

Code: `src/er/transliterate.py`, `src/er/normalize.py`. Tests: `tests/test_transliterate.py`, `tests/test_normalize.py`, `tests/test_normalize_properties.py`, `tests/test_real_data.py`.

## Why

The same business looks very different across sources. Real matched records from the training ground truth:

| Source 1 | Source 2 / 3 |
|---|---|
| `Clyial Pony LLC` | `L.L.C. Clyial Pony`, `Clyial Pony-LLC`, `LLC CLYIAL PONY` |
| `Agro India Private Limited` | `AGRO INDIA PRIVATE  LTD` |
| `Red Media Private Limited` | `रेड मीडिया प्राइवेट लिमिटेड` |
| `109 Jackson Street, Scott City, KS` | `00109 JACKSON ST, SCOTT CITY, KS` |
| `8008 14 Avenue, Brooklyn, NY` | `008008 14 Ave, Brooklyn, New York` |

Normalization removes the differences that never carry meaning (case, script, punctuation, abbreviations, suffix position, zero-padding) so that the later stages only have to deal with the hard ones (typos, missing words, different house numbers).

## Where it sits

```
raw records ──► AnyAsciiTransliterator ──► RuleNormalizer ──► blocking / matching
```

Both are classes with `transform(df) -> df`, selected by name in `main.py` (`--transliterator anyascii --normalizer rules`).

The normalizer **adds** columns and never changes the raw ones, so later stages can still look at the original text:

| Column | Example (from `Smt Gajanan Services Limited-Private`) |
|---|---|
| `name_norm` | `smt gajanan services` |
| `legal_form` | `ltd pvt` |
| `address_norm` | `409 marlborough street, 42, boston, ma` |

## Step 1: transliteration

Sources 2 and 3 contain nine Indic scripts (Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Odia, Tamil, Telugu), often mixed with Latin in one name. Source 1 is always Latin. `AnyAsciiTransliterator` converts everything to ASCII with the `anyascii` library:

| Input | Output |
|---|---|
| `ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `guru estet praivet limited` |
| `గుజరాత్ Logistics లిమిటెడ్` | `gujrat Logistics limited` |
| `SCI Ptit Àmicale` | `SCI Ptit Amicale` |

- Text that is already ASCII is returned without calling the library.
- It is lossy: some vowels are dropped (`gujrat`, `skti` for *shakti*), nasal marks become `m` (`marketimg`), and "private" comes out differently per script (`praivet`, `piraivet`, `praibhet`, `praivrr`). The normalizer's tables list these variants.

## Step 2: names

`normalize_name(name, country) -> (core, legal_form)`. Steps in order:

1. **Reduce the alphabet.** Lowercase, then replace everything except `a-z 0-9 & + . /` and spaces with a space. Every later rule sees the same characters the output is built from.
2. **Drop website parts.** `www.` and `.com/.net/.org/.in/.co.in/.fr`: `gajananservices.com` → `gajananservices`.
3. **Devanagari "Pvt. Ltd." abbreviation.** `pra. li.` → `pvt ltd`. A dot or space between the two parts is required, so `prali` is left alone.
4. **Dots.** Dots inside initials are removed so the letters join (`l.l.c.` → `llc`, `d.b.a.` → `dba`, `j.r.` → `jr`). Every other dot becomes a space (`pvt.ltd` → `pvt ltd`).
5. **DBA.** Split on `dba`, `d/b/a`, `trading as`, `t/a` and keep the **last** part that has letters or digits: `Fayewexhalo D.B.A. Precision-Telecom Dynamics, Inc` → `precision telecom dynamics`.
6. **Digits used as letters.** `0` → `o` and `1` → `l`, but only between two letters: `Federati0n` → `federation`. `A1 Bakery` and `3M` are untouched.
7. **`&` and `+` become `and`.**
8. **Tokenize** on anything that isn't `a-z0-9`, then map name words: `st` → `saint` everywhere; in France also `et` → `and` and `ets` → `etablissements`.
9. **Pull out legal forms.** Any token that is a legal form is removed from the core and its canonical form is added to `legal_form`, which is sorted and deduplicated. This works whatever the position: `Meighan P.C. Wheeler` → (`meighan wheeler`, `pc`), and `Bond Tecealom Pvt Pvt Ltd` → (`bond tecealom`, `ltd pvt`).
10. **Tidy up.** Remove a leading or trailing `and` left over from `Ss & Co`. If the name was *only* legal words (`Inc`), keep them as the core so the name isn't empty.

### Legal forms

| Canonical | Spellings |
|---|---|
| `pvt` | pvt, private, praivet, piraivet, praibhet, praivrr |
| `ltd` | ltd, limited, limitet, limirrd |
| `llc` `pllc` `llp` `lp` | same, plus `elelpi` (LLP spelled out in Devanagari) |
| `inc` | inc, incorporated |
| `corp` | corp, corporation |
| `co` | co, company (France also: cie, compagnie) |
| `pc` `opc` | same |
| France only | sarl, sas, sasu, sci, eurl, sa, ei, snc, selarl, scop, gie |

French forms apply only when `country == "France"`, because `SA`, `SAS` and `EI` are more likely initials in US or Indian names (`SAS Institute`).

Legal forms go in their own column instead of being deleted: the core name blocks better without them, but a mismatch (`LLC` vs `Pvt Ltd`) is still a useful signal for the matcher. On matched pairs where both sides have one, they agree 90% of the time.

### Examples

| Input (country) | `name_norm` | `legal_form` |
|---|---|---|
| `Fayewexhalo D.B.A. Precision-Telecom Dynamics, Inc` (US) | `precision telecom dynamics` | `inc` |
| `राम मार्केटिंग प्रा. लि.` (India) | `ram marketimg` | `ltd pvt` |
| `Montreal et Cie SARL` (France) | `montreal` | `co sarl` |
| `*** Ss Co  &` (India) | `ss` | `co` |
| `Empire Federati0n` (US) | `empire federation` | |

## Step 3: addresses

`normalize_address(address, country) -> str`. The address is split on commas, each **component** is normalized separately, and the result is re-joined with `", "`. Component order is kept, because sources reorder components (`TX, CLEVELAND, 1453 ROAD 5705`) and token-set comparisons downstream don't care about order.

For each component:

1. **Hyphens and slashes** are kept only between digits (`26/34`, `54-18-45`). Elsewhere they separate words (`Sector-3` → `sector 3`, `S/O` → `s o`).
2. **Clean** to `a-z 0-9 / -` tokens. `N°` (which transliterates to `ndeg`) becomes `no`.
3. **State names become codes.** The check runs on the **whole component** first: `Massachusetts` → `ma`, `महाराष्ट्र` (transliterated `mharastr`) → `mh`, `Loire-Atlantique` → `pdl`. Because only whole components are matched, `123 New York Avenue` is untouched. A component that is already a code (`CT`, `FL`) is kept as-is and never expanded to `court` or `floor`.
4. **Expand street words** (`rd` → `road`, `ave` → `avenue`, `ct` → `court`, …), strip leading zeros from pure numbers (`003160` → `3160`), and drop filler words (`unit`, `suite`, `city`, `City of …`).
   - `st` depends on context. It means **street** when it ends the component or is followed by a number, a single letter, a direction or a unit word (`Elm St`, `Elm St NW`, `Elm St Apt 5`). Otherwise it means **saint** (`St Louis`, `Rue St Pierre`, `ST-NAZAIRE`).
   - `ste` works the same way: it's a **suite** (dropped) before a number or single letter (`Ste 200`), otherwise **sainte** (`Ste-Foy`).
   - **Kansas City guard:** if dropping filler would leave a bare state name, the component is rebuilt without dropping anything. So `Kansas City` stays `kansas city` and isn't read as the state Kansas.
5. **Drop null placeholders**: `null`, `<NULL>`, `N/A`, `none`.

Finally, **repeated components are removed** (keeping the first), so `Mumbai, Bombay` and `Kolkata, Kolkata` collapse to one.

### Country-specific rules

These are chosen by the record's `country` column. Any other country, including labels unseen in training, gets only the generic rules.

| Country | Rule |
|---|---|
| France | `r` → rue, `imp` → impasse, `all` → allee, `chem` → chemin, `rte` → route, `fg` → faubourg, `crs` → cours, `qu` → quai |
| France | Départements map to their région, so both levels agree: `Nord` / `Pas-de-Calais` → `hdf`, `Gironde` → `naq`, `Loire-Atlantique` → `pdl` |
| India | Old or alternate city names: Bombay → mumbai, Calcutta → kolkata, Madras → chennai, Bangalore → bengaluru, Gurgaon → gurugram, Ahilyanagar → ahmednagar, … |

State tables: all 50 US states + DC, the Indian states in English plus their transliterated native-script forms (`dilli`, `krnatk`, `pscimbng`, …), and the three French régions in the test data with their départements. Full state names are unique across countries, so they share one table.

### Examples

| Input (country) | `address_norm` |
|---|---|
| `0409 Marlborough St, Unit 42, <NULL>, Boston, Massachusetts` (US) | `409 marlborough street, 42, boston, ma` |
| `12 Asylum St, Hartford, CT` (US) | `12 asylum street, hartford, ct` |
| `Kansas City, KS` (US) | `kansas city, ks` |
| `Office 1, Mumbai, Bombay, महाराष्ट्र` (India) | `office 1, mumbai, mh` |
| `13 R. DU PLESSIS, ST-NAZAIRE, Loire-Atlantique` (France) | `13 rue du plessis, saint nazaire, pdl` |

## Guarantees

Checked by the tests on every run:

- **Output alphabet:** `name_norm` is `[a-z0-9 ]` only. Address tokens are `[a-z0-9]` joined by `/` or `-`. No double or outer spaces. Output is always ASCII after transliteration.
- **Idempotent:** normalizing the output again changes nothing. This held on 1.8M real rows.
- **Raw columns and the index are untouched**, and the input DataFrame is never mutated.
- **Legal forms are canonical**, sorted and unique.

## Validation

Measured on 73,333 matched pairs from the training ground truth, against 73,333 random non-matching pairs from the same country. The baseline is transliterate + lowercase + strip punctuation.

| | Baseline | Normalized |
|---|---|---|
| Matches: names exactly equal | 25.8% | **51.9%** |
| Matches: addresses exactly equal | 8.4% | **27.8%** |
| Matches: address token Jaccard | 0.607 | **0.790** |
| Non-matches: names exactly equal | 0.0% | 0.0% |
| Non-matches: name token Jaccard | 0.030 | **0.003** |

True matches get much closer. Random non-matches don't collide, and they share fewer name tokens because common filler like "Pvt Ltd" no longer counts.

Speed: about 50k rows/s, so all ~22M records take about 7 minutes.

## Known limitations

- **Typos** are not fixed: `Propoerttes`, `Pibvate`, `KYASVILLE`. That's left to fuzzy matching.
- **Transliteration quirks** remain: `marketimg`, `skti`. Common Indic business words come out mangled (`imtrnesnl` = international). A word table mined from the ground truth would fix the frequent ones.
- **Letter-for-digit swaps** aren't covered: `lndia` (l for I).
- **Parenthesised country names** such as `(India)` are **kept**, because 79% of matched records keep them. That makes them part of the name, not noise.
- **Different levels of place names** can't be reconciled: `Chesterfield County` vs `Richmond`, and `New Delhi` vs `West Delhi`.
- **`Washington` or `Delhi` as a city component** is read as the state, which is harmless because it happens the same way on both sides.
- **The non-match check uses random pairs**, which are easy to tell apart. Hard negatives (same street, similar name) will be measured once blocking exists.

## Adding a rule

1. Find the pattern in real data. The best source is matched groups from `train_ground_truth.tsv` whose normalized forms still differ.
2. Add the rule or table entry in `src/er/normalize.py`. Country-specific rules go in the `COUNTRY_*` dicts.
3. Add parametrized cases to `tests/test_normalize.py` using the real strings, including a case where the rule must **not** fire.
4. Run `uv run pytest`. The property tests will catch rules that break idempotence or the output alphabet. The most common cause is a rule that runs before punctuation is stripped.
