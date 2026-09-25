"""Text cleaning for business entity resolution.

Pure, deterministic str -> str functions. No I/O, no external lookups, no pandas.
Every public function accepts any input (None, NaN, numbers, any Unicode) and never raises.

Entry point: `clean_record`. It normalizes `country` once; every function below it
takes an already-cleaned (lowercase) country and never re-normalizes it.
"""

import math
import re
import unicodedata

from anyascii import anyascii

# Whole-field values that mean "missing".
NULL_VALUES = {"", "<null>", "null", "nan", "n/a"}

# ---------------------------------------------------------------- names

# Alias markers, matched CASE-SENSITIVELY on the raw name. Ground truth: the matching
# name is the part AFTER the marker (fka 7,490/7,491, dba 31,411/31,422, ...).
# "Aka", "AKA", "Dba" are real words in names ("Aka Solution Ltd") and are not markers.
ALIAS_MARKER = re.compile(
    r"(?<!\S)(?:formerly known as|formerly|Formerly|a/k/a|aka|d/b/a|dba|DBA|f/k/a|F/K/A|fka|FKA)(?!\S)"
)
MS_PREFIX = re.compile(r"^\s*m/s\b\.?", re.I)  # "M/s Haven Exim" (Messrs)
WEBSITE = re.compile(r"\b(?:https?://)?www\.|\.(?:com|net|org|co\.in|in|fr)\b")
INITIALS = re.compile(r"\b[a-z](?:\.[a-z])+\.?(?![a-z])")  # "l.l.c." -> "llc", "p.c." -> "pc"
LEET = [(re.compile(r"(?<=[a-z])0(?=[a-z])"), "o"), (re.compile(r"(?<=[a-z])1(?=[a-z])"), "l")]

# Legal forms -> short canonical token (decision D1: keep short). Includes the anyascii
# spellings of the Indic forms seen in the data (e.g. प्राइवेट -> "praivet", एलएलपी -> "elelpi").
LEGAL_FORMS = {
    "company": "co",
    "corporation": "corp",
    "elelpi": "llp",
    "incorporated": "inc",
    "limirrd": "ltd", "limited": "ltd", "limitet": "ltd",
    "piraivet": "pvt", "praibhet": "pvt", "praivet": "pvt", "praivrr": "pvt", "private": "pvt",
}
# Forms that are only legal suffixes in one country ("SA" elsewhere is more likely initials).
COUNTRY_LEGAL_FORMS = {
    "france": {"sarl": "sarl", "sas": "sas", "sasu": "sasu", "sci": "sci", "eurl": "eurl", "sa": "sa",
               "ei": "ei", "snc": "snc", "selarl": "selarl", "scop": "scop", "gie": "gie",
               "cie": "co", "compagnie": "co"},
}
NAME_WORDS = {"st": "saint"}  # in a business name "St" is Saint
COUNTRY_NAME_WORDS = {"france": {"et": "and", "ets": "etablissements"}}  # French "et" is "&"

# ---------------------------------------------------------------- addresses

# Mumbai-style suffixes, rewritten on the raw string BEFORE comma splitting because the
# garbled form "(We, St)" contains a comma.
DIRECTION_SUFFIXES = [
    (re.compile(r"\(\s*we\s*,\s*st\s*\)"), " west "),
    (re.compile(r"\(\s*ea\s*,\s*st\s*\)"), " east "),
    (re.compile(r"\(\s*(?:w|west)\s*\)"), " west "),
    (re.compile(r"\(\s*(?:e|east)\s*\)"), " east "),
]
# Fragments left when a source reordered the parts of "(Ea, St)" ("Bandra (Ea, MUMBAI, ST)").
ORPHAN_CLOSE = re.compile(r"^\s*st\s*\)\s*$")
ORPHAN_OPEN = re.compile(r"\(\s*(ea|we)\s*$")

STREET_WORDS = {
    "apt": "apartment", "apts": "apartments", "av": "avenue", "ave": "avenue",
    "bd": "boulevard", "bldg": "building", "blk": "block", "blvd": "boulevard",
    "cir": "circle", "ct": "court", "dist": "district", "extn": "extension",
    "fl": "floor", "flr": "floor", "hwy": "highway", "ln": "lane", "nr": "near",
    "opp": "opp", "opposite": "opp",  # decision 3a: keep "opp" short (Opp, AL is a city)
    "pkwy": "parkway", "pl": "place", "rd": "road", "soc": "society", "sq": "square",
    "ter": "terrace", "trl": "trail",
}
# Extra street words that only apply to one country ("R." is "rue" in France, not elsewhere).
COUNTRY_STREET_WORDS = {
    "france": {"r": "rue", "all": "allee", "imp": "impasse", "chem": "chemin", "rte": "route",
               "fg": "faubourg", "crs": "cours", "qu": "quai"},
}
# Old/alternate Indian city names; sources mix them ("Mumbai, Bombay", "Kolkata, Calcutta").
# Applied only to a whole comma part, never inside one ("Opp. Bank Of Baroda" is a landmark).
# The old names are unambiguous as whole parts, so this applies regardless of country.
CITY_ALIASES = {
    "ahilyanagar": "ahmednagar", "ahmadabad": "ahmedabad", "allahabad": "prayagraj",
    "bangalore": "bengaluru", "baroda": "vadodara", "belgaum": "belagavi", "bombay": "mumbai",
    "calcutta": "kolkata", "calicut": "kozhikode", "cochin": "kochi", "gurgaon": "gurugram",
    "kancheepuram": "kanchipuram", "madras": "chennai", "malapuram": "malappuram",
    "mysore": "mysuru", "newdelhi": "new delhi", "poona": "pune", "sonepat": "sonipat",
    "thiruvallur": "tiruvallur", "trivandrum": "thiruvananthapuram",
    "vishakhapatnam": "visakhapatnam",
}
# After these, "st"/"dr"/"ste" are street types ("Elm St NW"); before anything else
# they are Saint / Doctor / Sainte ("St Louis", "Dr G D Marg").
STREET_TYPE_FOLLOWERS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west",
                         "apt", "apartment", "unit", "suite", "ste", "fl", "floor", "flr", "bldg"}
ADDRESS_DROP = {"unit", "suite"}  # filler that differs between sources for one address
# "Ogden City" / "Mumbai City" -> "ogden" / "mumbai"; a "city" mid-part is a name ("Lake City Mall").
TRAILING_CITY = re.compile(r"(?: city)+$")
CITY_OF = re.compile(r"^(?:(?:city|town|village) )?of (?=\S)")  # "City of El Paso" -> "el paso"
NUMERO = re.compile(r"\bn\s*°\s*", re.I)  # "N°157" (anyascii would give "ndeg157") -> "no 157"
NULL_PARTS = {"null", "na", "n a", "none", "nan"}  # "<NULL>" / "N/A" after cleaning

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
# English names plus the anyascii output of the native-script names seen in the data.
INDIA_STATES = {
    "andhra pradesh": "ap", "amdhrprdes": "ap", "assam": "as", "bihar": "br",
    "chandigarh": "ch", "chhattisgarh": "cg", "delhi": "dl", "dilli": "dl", "goa": "ga",
    "gujarat": "gj", "gujrat": "gj", "haryana": "hr", "hriyana": "hr",
    "himachal pradesh": "hp", "jammu and kashmir": "jk", "jharkhand": "jh",
    "karnataka": "ka", "krnatk": "ka", "kerala": "kl", "keralam": "kl", "kerlm": "kl",
    "madhya pradesh": "mp", "mdhy prdes": "mp", "maharashtra": "mh", "mharastr": "mh",
    "odisha": "od", "od isa": "od", "orissa": "od", "puducherry": "py", "punjab": "pb",
    "pmjab": "pb", "rajasthan": "rj", "rajsthan": "rj", "tamil nadu": "tn", "tmilnatu": "tn",
    "telangana": "tg", "telmgan": "tg", "uttar pradesh": "up", "uttr prdes": "up",
    "uttarakhand": "uk", "west bengal": "wb", "pscimbng": "wb",
}
# French regions in the data, with their departements mapped to the region.
FRANCE_REGIONS = {
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf", "aisne": "hdf", "oise": "hdf",
    "somme": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq", "charente": "naq", "charente maritime": "naq",
    "correze": "naq", "creuse": "naq", "dordogne": "naq", "landes": "naq", "lot et garonne": "naq",
    "pyrenees atlantiques": "naq", "deux sevres": "naq", "vienne": "naq", "haute vienne": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl", "maine et loire": "pdl", "mayenne": "pdl",
    "sarthe": "pdl", "vendee": "pdl",
}
# Full state names are unique across countries, so one table is safe for any country label.
STATES = US_STATES | INDIA_STATES | FRANCE_REGIONS
STATE_CODES = set(STATES.values())

# Postal codes, only where the shape is unmistakable (decision D4). Keys are cleaned
# (lowercase) country values. Indian PINs never start with 0, which rules out the
# zero-padded house numbers common in the data ("008006 Rolling Oaks Dr").
POSTAL_AT_END = {
    "india": re.compile(r"(?:^|[\s,])([1-9]\d{5})\s*$"),
    "us": re.compile(r"(?:^|[\s,])(\d{5}(?:-\d{4})?)\s*$"),
}
POSTAL_AFTER_PIN = {
    "india": re.compile(r"\bpin(?:\s*code)?\s*[:.\-]?\s*([1-9]\d{5})\b"),
}


# ---------------------------------------------------------------- shared helpers

def _to_text(value) -> str:
    """Any input -> str. None / NaN / null markers -> ""."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = value if isinstance(value, str) else str(value)
    return "" if text.strip().lower() in NULL_VALUES else text


def _to_ascii(text: str) -> str:
    """NFKC, then transliterate every script (Devanagari, Telugu, accented Latin, ...) to ASCII."""
    text = unicodedata.normalize("NFKC", text)
    return text if text.isascii() else anyascii(text)


def _tokens(text: str, keep: str = "") -> list[str]:
    """Replace every char outside [a-z0-9] + `keep` with a space, split."""
    return re.sub(rf"[^a-z0-9{keep}]+", " ", text).split()


def _dedupe_adjacent(items: list[str]) -> list[str]:
    return [x for i, x in enumerate(items) if i == 0 or x != items[i - 1]]


def _non_null(text: str) -> str:
    """A cleaned value that reads as a null marker ("nan", "null") is empty, so cleaning is idempotent."""
    return "" if text in NULL_VALUES else text


# ---------------------------------------------------------------- country

def clean_country(country) -> str:
    """Free-text country label -> lowercase ASCII. Open set: no fixed list of countries."""
    return _non_null(" ".join(_tokens(_to_ascii(_to_text(country)).lower())))


# ---------------------------------------------------------------- name

def resolve_alias(name: str) -> str:
    """'X fka Y' / 'X d/b/a Y' -> 'Y'. Case-sensitive markers; no split when nothing precedes
    the marker ('DBA CONSTRUCTION') or nothing follows it."""
    m = ALIAS_MARKER.search(name)
    if not m:
        return name
    before, after = name[:m.start()], name[m.end():]
    if not re.search(r"\w", before) or not re.search(r"\w", after):
        return name
    return after


def clean_name(name, country: str = "") -> str:
    """Raw business name -> cleaned name. Legal forms are kept in place, in short form.

    `country` must already be clean (lowercase), as produced by `clean_country`.
    """
    s = MS_PREFIX.sub(" ", _to_ascii(_to_text(name)))  # before aliases: "M/s DBA X" has no alias
    s = resolve_alias(s).lower()  # before lowercasing: markers are case-sensitive
    s = WEBSITE.sub(" ", s)
    s = INITIALS.sub(lambda m: m.group().replace(".", ""), s)
    for pattern, repl in LEET:
        s = pattern.sub(repl, s)
    s = s.replace("&", " and ").replace("+", " and ")
    tokens = _tokens(s)
    # Devanagari "प्रा. लि." transliterates to "pra li".
    out = []
    for tok in tokens:
        if tok == "li" and out and out[-1] == "pra":
            out[-1:] = ["pvt", "ltd"]
        else:
            out.append(tok)
    words = NAME_WORDS | COUNTRY_NAME_WORDS.get(country, {})
    legal_forms = LEGAL_FORMS | COUNTRY_LEGAL_FORMS.get(country, {})
    out = [words.get(t, legal_forms.get(t, t)) for t in out]
    return _non_null(" ".join(_dedupe_adjacent(out)))


# ---------------------------------------------------------------- address

def _clean_part(part: str, street_words: dict[str, str], drop_filler: bool = True) -> str:
    """One comma-separated address part -> cleaned part."""
    m = ORPHAN_OPEN.search(part)
    if m:  # "bandra (ea" -> "bandra east" (the rest of "(Ea, St)" went to another part)
        part = part[:m.start()] + (" east" if m.group(1) == "ea" else " west")
    # Keep "/" and "-" only inside numbers ("26/34", "54-18-45"); elsewhere they separate words.
    part = re.sub(r"(?<![0-9])[-/]|[-/](?![0-9])", " ", part)
    # Drop filler first, so the St/Dr context below sees the same neighbours on every pass.
    tokens = [t for t in _tokens(part, keep="/-") if not (drop_filler and t in ADDRESS_DROP)]
    out = []
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        street_type = nxt is None or nxt in STREET_TYPE_FOLLOWERS or nxt[0].isdigit() or len(nxt) == 1
        if tok == "st":
            out.append("street" if street_type else "saint")
        elif tok == "dr":  # "Dr G D Marg" is Doctor: initials don't make it a street type
            out.append("drive" if nxt is None or nxt in STREET_TYPE_FOLLOWERS or nxt[0].isdigit() else "dr")
        elif tok == "ste":  # "Ste 200" is a suite (filler), "Ste Foy" is Sainte
            if not street_type:
                out.append("sainte")
            elif not drop_filler:
                out.append("suite")
        elif tok.isdigit():
            out.append(tok.lstrip("0") or "0")
        else:
            out.append(street_words.get(tok, tok))
    # "Fl 0" / "Floor 0" is a placeholder, not a real floor (decision 3d).
    i = 0
    while i < len(out) - 1:
        if out[i] == "floor" and out[i + 1] == "0":
            del out[i:i + 2]
            i = max(i - 1, 0)
        else:
            i += 1
    s = " ".join(out)
    s = CITY_OF.sub("", TRAILING_CITY.sub("", s)) if drop_filler else s
    return CITY_ALIASES.get(s, s)


def _address_parts(address, country: str = "") -> list[str]:
    """Raw address + CLEANED country -> list of cleaned, non-empty comma parts."""
    street_words = STREET_WORDS | COUNTRY_STREET_WORDS.get(country, {})
    s = _to_ascii(NUMERO.sub("no ", unicodedata.normalize("NFKC", _to_text(address)))).lower()
    for pattern, repl in DIRECTION_SUFFIXES:
        s = pattern.sub(repl, s)
    parts = []
    for raw in s.split(","):
        if ORPHAN_CLOSE.match(raw):
            continue
        plain = " ".join(_tokens(raw))
        if plain in STATES or plain in STATE_CODES:  # "CT"/"FL" alone are states, not court/floor
            parts.append(plain)
            continue
        part = _clean_part(raw, street_words)
        if part in STATES:  # dropping filler produced a state name ("Kansas City"): keep the filler
            part = _clean_part(raw, street_words, drop_filler=False)
        if part and part not in NULL_PARTS:
            parts.append(part)
    return parts


def _state_positions(parts: list[str], country: str) -> list[int]:
    """Indexes where a state may appear: the last part, and parts next to a country part
    (decision 3b — so 'Washington, DC' keeps Washington as the city)."""
    positions = {len(parts) - 1} if parts else set()
    for i, p in enumerate(parts):
        if country and p == country:
            positions |= {i - 1, i + 1}
    return sorted(i for i in positions if 0 <= i < len(parts) and parts[i] != country)


def _dedupe_keep_last(parts: list[str]) -> list[str]:
    """Drop repeated parts, keeping the LAST occurrence so the final part never changes."""
    seen, out = set(), []
    for p in reversed(parts):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[::-1]


def parse_address(address, country: str = "") -> dict:
    """Raw address + CLEANED country -> {address, postal_code, state, city, country}.

    `country` must already be clean (lowercase), as produced by `clean_country`.
    """
    # Convert state names to codes at the allowed positions, then drop repeated parts.
    # Dropping a part can move a state name next to the country, so repeat until stable.
    parts = _dedupe_keep_last(_address_parts(address, country))
    while True:
        converted = [STATES.get(p, p) if i in _state_positions(parts, country) else p
                     for i, p in enumerate(parts)]
        converted = _dedupe_keep_last(converted)
        if converted == parts:
            break
        parts = converted
    state_idx = next((i for i in _state_positions(parts, country) if parts[i] in STATE_CODES), None)
    state = parts[state_idx] if state_idx is not None else ""
    # City: nearest digit-free part before the state (skipping the country part).
    city = ""
    if state_idx is not None:
        for p in reversed(parts[:state_idx]):
            if p != country and not re.search(r"\d", p):
                city = p
                break
    return {
        "address": ", ".join(parts),
        "postal_code": _postal_code(address, country),
        "state": state,
        "city": city,
        "country": country,
    }


def _postal_code(address, country: str) -> str:
    """Postal code only when the shape is unmistakable for this (cleaned) country; else ""."""
    s = _to_ascii(_to_text(address)).lower()
    for table in (POSTAL_AFTER_PIN, POSTAL_AT_END):
        pattern = table.get(country)
        m = pattern.search(s) if pattern else None
        if m:
            return m.group(1)
    return ""


def clean_address(address, country: str = "") -> str:
    """Raw address -> full cleaned address string. `country` must already be clean."""
    return parse_address(address, country)["address"]


# ---------------------------------------------------------------- entry point

def clean_record(record: dict) -> dict:
    """Raw record -> raw fields plus cleaned ones. Country is normalized here, once."""
    country = clean_country(record.get("country"))
    parsed = parse_address(record.get("business_address"), country)
    return {
        **record,
        "name_clean": clean_name(record.get("business_name"), country),
        "address_clean": parsed["address"],
        "postal_code": parsed["postal_code"],
        "state": parsed["state"],
        "city": parsed["city"],
        "country_clean": country,
    }
