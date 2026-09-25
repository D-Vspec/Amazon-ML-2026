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
}
# Forms that are only legal suffixes in one country ("SA" elsewhere is more likely initials).
COUNTRY_LEGAL_FORMS = {
    "France": {"sarl": "sarl", "sas": "sas", "sasu": "sasu", "sci": "sci", "eurl": "eurl", "sa": "sa",
               "ei": "ei", "snc": "snc", "selarl": "selarl", "scop": "scop", "gie": "gie",
               "cie": "co", "compagnie": "co"},
}
# Name word variants: "St" in a business name is Saint; French "et" is "&".
NAME_WORDS = {"st": "saint"}
COUNTRY_NAME_WORDS = {"France": {"et": "and", "ets": "etablissements"}}

# "X dba Y" keeps only the trade name Y (the last part that has any letters/digits).
DBA = re.compile(r"\b(?:dba|d/b/a|trading as|t/a)\b")
WEBSITE = re.compile(r"^www\.|\.(?:com|net|org|in|co\.in|fr)\b")
LEET = [(re.compile(r"(?<=[a-z])0(?=[a-z])"), "o"), (re.compile(r"(?<=[a-z])1(?=[a-z])"), "l")]
PVT_LTD_ABBR = re.compile(r"\bpra(?:\.\s*|\s+)li\b\.?")  # Devanagari "प्रा. लि."
INITIALS = re.compile(r"\b[a-z](?:\.[a-z])+\.?(?![a-z])")  # "l.l.c.", "d.b.a", "j.r."

STREET_WORDS = {
    "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane",
    "ct": "court", "pl": "place", "cir": "circle", "blvd": "boulevard", "bd": "boulevard",
    "hwy": "highway", "pkwy": "parkway", "ter": "terrace", "trl": "trail", "sq": "square",
    "fl": "floor", "flr": "floor", "bldg": "building", "apt": "apartment", "apts": "apartments",
    "opp": "opposite", "nr": "near", "dist": "district",
}
# Extra street words that only apply to one country ("R." is "rue" in France, not elsewhere).
COUNTRY_STREET_WORDS = {
    "France": {"r": "rue", "all": "allee", "imp": "impasse", "chem": "chemin", "rte": "route",
               "fg": "faubourg", "crs": "cours", "qu": "quai"},
    # Old/alternate city names; sources mix them ("Mumbai, Bombay", "Kolkata, Calcutta").
    "India": {"bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bangalore": "bengaluru",
              "poona": "pune", "gurgaon": "gurugram", "cochin": "kochi", "calicut": "kozhikode",
              "trivandrum": "thiruvananthapuram", "ahilyanagar": "ahmednagar", "allahabad": "prayagraj",
              "mysore": "mysuru", "belgaum": "belagavi", "baroda": "vadodara", "ahmadabad": "ahmedabad",
              "vishakhapatnam": "visakhapatnam", "sonepat": "sonipat", "kancheepuram": "kanchipuram",
              "malapuram": "malappuram", "thiruvallur": "tiruvallur", "newdelhi": "new delhi"},
}
# "City of El Paso" -> "el paso" ("city" alone is already dropped, leaving "of el paso").
CITY_OF = re.compile(r"^(?:(?:city|town|village) )?of (?=\S)")
# "N°" transliterates to "ndeg".
NUMERO = re.compile(r"\bndeg(?=\d|\s|$)")
# Unit markers and filler that differ between sources for the same address.
ADDRESS_DROP = {"unit", "suite", "city"}
# After these, "st" is a street type ("Elm St NW", "Elm St Apt 5"); before anything else it is "saint".
STREET_TYPE_FOLLOWERS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west",
                         "apt", "apartment", "unit", "suite", "ste", "fl", "floor", "flr", "bldg"}
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
    "kerala": "kl", "keralam": "kl", "kerlm": "kl", "bihar": "br", "madhya pradesh": "mp", "mdhy prdes": "mp",
    "andhra pradesh": "ap", "amdhrprdes": "ap", "punjab": "pb", "pmjab": "pb",
    "odisha": "od", "orissa": "od", "od isa": "od", "goa": "ga", "assam": "as",
    "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "himachal pradesh": "hp",
    "jammu and kashmir": "jk", "puducherry": "py", "chandigarh": "ch",
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


def _clean(text: str, keep: str) -> list[str]:
    """Lowercase, replace every char outside [a-z0-9] + `keep` with a space, split."""
    return re.sub(rf"[^a-z0-9{keep}]+", " ", text.lower()).split()


class RuleNormalizer:
    def normalize_name(self, name: str, country: str = "") -> tuple[str, str]:
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
        words = NAME_WORDS | COUNTRY_NAME_WORDS.get(country, {})
        tokens = [words.get(tok, tok) for tok in _clean(s, keep="")]

        legal_forms = LEGAL_FORMS | COUNTRY_LEGAL_FORMS.get(country, {})
        core = [tok for tok in tokens if tok not in legal_forms]
        legal = sorted({legal_forms[tok] for tok in tokens if tok in legal_forms})
        # "Ss & Co" -> drop the dangling "and" left behind by the legal form.
        while core and core[-1] == "and":
            core.pop()
        while core and core[0] == "and":
            core.pop(0)
        if not core and legal:  # name was only legal words, keep them as the name
            core = [legal_forms.get(tok, tok) for tok in tokens]
        return " ".join(core), " ".join(legal)

    def normalize_address(self, address: str, country: str = "") -> str:
        """Normalize each comma-separated component; drop null ones. Order is preserved."""
        street_words = STREET_WORDS | COUNTRY_STREET_WORDS.get(country, {})
        components = []
        for part in address.lower().split(","):
            # Keep "/" and "-" only inside numbers ("26/34", "54-18-45"); elsewhere they separate words.
            part = re.sub(r"(?<![0-9])[-/]|[-/](?![0-9])", " ", part)
            tokens = NUMERO.sub("no ", " ".join(_clean(part, keep="/-"))).split()
            # State lookup on the whole component first, so "Kansas City" is not read as "Kansas".
            component = " ".join(tokens)
            if component in STATES:
                components.append(STATES[component])
                continue
            if component in STATE_CODES:  # "CT"/"FL" are states here, not court/floor
                components.append(component)
                continue
            component = self._component(tokens, street_words, drop_filler=True)
            if component in STATES:  # dropping filler left a state name ("Kansas City"): keep the filler
                component = self._component(tokens, street_words, drop_filler=False)
            if component and component not in NULL_COMPONENTS:
                components.append(component)
        return ", ".join(dict.fromkeys(components))  # drop repeats ("Kolkata, Kolkata"), keep order

    @staticmethod
    def _component(tokens: list[str], street_words: dict[str, str], drop_filler: bool) -> str:
        """Expand street words, strip leading zeros and (optionally) drop unit/city filler."""
        out = []
        for i, tok in enumerate(tokens):
            if drop_filler and tok in ADDRESS_DROP:
                continue
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            street_type = nxt is None or nxt in STREET_TYPE_FOLLOWERS or nxt[0].isdigit() or len(nxt) == 1
            if tok == "st":
                out.append("street" if street_type else "saint")
            elif tok == "ste":  # "Ste 200" / "Ste B" is a suite (filler), "Ste-Foy" is Sainte
                if not street_type:
                    out.append("sainte")
                elif not drop_filler:
                    out.append("suite")
            elif tok.isdigit():
                out.append(tok.lstrip("0") or "0")
            else:
                out.append(street_words.get(tok, tok))
        component = " ".join(out)
        return CITY_OF.sub("", component) if drop_filler else component

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        records = records.copy()
        names = [self.normalize_name(nm, c) for nm, c in zip(records["business_name"], records["country"])]
        records["name_norm"] = [core for core, _ in names]
        records["legal_form"] = [legal for _, legal in names]
        records["address_norm"] = [self.normalize_address(a, c)
                                   for a, c in zip(records["business_address"], records["country"])]
        return records
