"""Rule-based cleanup of (already transliterated) names and addresses.

Adds `name_norm`, `legal_form` and `address_norm` columns; raw columns are left untouched.
"""

import re

import pandas as pd

# Legal-form spellings (incl. transliterated Indic ones) -> canonical token.
LEGAL_FORMS = {
    "pvt": "pvt", "private": "pvt", "praivet": "pvt", "piraivet": "pvt", "praibhet": "pvt", "praivrr": "pvt",
    "ltd": "ltd", "limited": "ltd", "limitet": "ltd", "limirrd": "ltd",
    "llc": "llc", "pllc": "pllc", "llp": "llp", "elelpi": "llp", "lp": "lp",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "pc": "pc", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sci": "sci", "eurl": "eurl",
}

# "X dba Y" keeps only the trade name Y (the last part that has any letters/digits).
DBA = re.compile(r"\b(?:dba|d/b/a|trading as|t/a)\b")
WEBSITE = re.compile(r"^www\.|\.(?:com|net|org|in|co\.in|fr)\b")
LEET = [(re.compile(r"(?<=[a-z])0(?=[a-z])"), "o"), (re.compile(r"(?<=[a-z])1(?=[a-z])"), "l")]
PVT_LTD_ABBR = re.compile(r"\bpra(?:\.\s*|\s+)li\b\.?")  # Devanagari "प्रा. लि."
INITIALS = re.compile(r"\b[a-z](?:\.[a-z])+\.?(?![a-z])")  # "l.l.c.", "d.b.a", "j.r."

STREET_WORDS = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane",
    "ct": "court", "pl": "place", "cir": "circle", "blvd": "boulevard", "bd": "boulevard",
    "hwy": "highway", "pkwy": "parkway", "ter": "terrace", "trl": "trail", "sq": "square",
    "fl": "floor", "flr": "floor", "bldg": "building", "apt": "apartment", "apts": "apartments",
    "opp": "opposite", "nr": "near", "dist": "district",
}
# Unit markers and filler that differ between sources for the same address.
ADDRESS_DROP = {"unit", "ste", "suite", "city"}
# Placeholders for a missing component, as they look after cleaning ("<NULL>" -> "null", "N/A" -> "n a").
NULL_COMPONENTS = {"null", "na", "n a", "none"}

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
    "maharashtra": "mh", "mharastr": "mh", "delhi": "dl", "dilli": "dl",
    "uttar pradesh": "up", "uttr prdes": "up", "karnataka": "ka", "krnatk": "ka",
    "tamil nadu": "tn", "tmilnatu": "tn", "gujarat": "gj", "gujrat": "gj",
    "west bengal": "wb", "pscimbng": "wb", "telangana": "tg", "telmgan": "tg",
    "haryana": "hr", "hriyana": "hr", "rajasthan": "rj", "rajsthan": "rj",
    "kerala": "kl", "kerlm": "kl", "bihar": "br", "madhya pradesh": "mp", "mdhy prdes": "mp",
    "andhra pradesh": "ap", "amdhrprdes": "ap", "punjab": "pb", "pmjab": "pb",
    "odisha": "od", "orissa": "od", "od isa": "od", "goa": "ga", "assam": "as",
    "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "himachal pradesh": "hp",
    "jammu and kashmir": "jk", "puducherry": "py", "chandigarh": "ch",
}
# Full state names are unique across countries, so one table is safe for any country label.
STATES = US_STATES | INDIA_STATES
STATE_CODES = set(STATES.values())


def _clean(text: str, keep: str) -> list[str]:
    """Lowercase, replace every char outside [a-z0-9] + `keep` with a space, split."""
    return re.sub(rf"[^a-z0-9{keep}]+", " ", text.lower()).split()


class RuleNormalizer:
    def normalize_name(self, name: str) -> tuple[str, str]:
        """Return (core name, legal form). Legal-form tokens are pulled out, sorted and deduplicated."""
        # Reduce to the output alphabet up front, keeping only the punctuation the rules below need.
        s = re.sub(r"[^a-z0-9&+./ ]+", " ", name.lower())
        s = WEBSITE.sub(" ", s)
        s = PVT_LTD_ABBR.sub(" pvt ltd ", s)
        # Join initials ("l.l.c." -> "llc"); any other dot separates words ("pvt.ltd" -> "pvt ltd").
        s = INITIALS.sub(lambda m: m.group().replace(".", ""), s).replace(".", " ")
        s = next((part for part in reversed(DBA.split(s)) if re.search(r"[a-z0-9]", part)), "")
        for pattern, repl in LEET:
            s = pattern.sub(repl, s)
        s = s.replace("&", " and ").replace("+", " and ")
        tokens = _clean(s, keep="")

        core = [tok for tok in tokens if tok not in LEGAL_FORMS]
        legal = sorted({LEGAL_FORMS[tok] for tok in tokens if tok in LEGAL_FORMS})
        # "Ss & Co" -> drop the dangling "and" left behind by the legal form.
        while core and core[-1] == "and":
            core.pop()
        while core and core[0] == "and":
            core.pop(0)
        if not core and legal:  # name was only legal words, keep them as the name
            core = [LEGAL_FORMS.get(tok, tok) for tok in tokens]
        return " ".join(core), " ".join(legal)

    def normalize_address(self, address: str) -> str:
        """Normalize each comma-separated component; drop null ones. Order is preserved."""
        components = []
        for part in address.lower().split(","):
            # Keep "/" and "-" only inside numbers ("26/34", "54-18-45"); elsewhere they separate words.
            part = re.sub(r"(?<![0-9])[-/]|[-/](?![0-9])", " ", part)
            tokens = _clean(part, keep="/-")
            # State lookup on the whole component first, so "Kansas City" is not read as "Kansas".
            component = " ".join(tokens)
            if component in STATES:
                components.append(STATES[component])
                continue
            if component in STATE_CODES:  # "CT"/"FL" are states here, not court/floor
                components.append(component)
                continue
            kept = [tok for tok in tokens if tok not in ADDRESS_DROP]
            if " ".join(kept) not in STATES:  # "Kansas City" keeps "city", else it reads as a state
                tokens = kept
            tokens = [STREET_WORDS.get(tok, tok) for tok in tokens]
            tokens = [tok.lstrip("0") or "0" if tok.isdigit() else tok for tok in tokens]
            component = " ".join(tokens)
            if component and component not in NULL_COMPONENTS:
                components.append(component)
        return ", ".join(components)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        records = records.copy()
        names = records["business_name"].map(self.normalize_name)
        records["name_norm"] = names.str[0]
        records["legal_form"] = names.str[1]
        records["address_norm"] = records["business_address"].map(self.normalize_address)
        return records
