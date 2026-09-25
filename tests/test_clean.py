"""Tests for src/clean.py. Examples are real records from the train sources unless noted."""

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import clean
from clean import (ALIAS_MARKER, POSTAL_AFTER_PIN, POSTAL_AT_END, clean_address, clean_country,
                   clean_name, clean_record, parse_address, resolve_alias)


# ---------------------------------------------------------------- nulls / base

@pytest.mark.parametrize("raw", [None, math.nan, float("nan"), "", "   ", "<NULL>", "null", "NaN", "N/A"])
def test_null_inputs_become_empty(raw):
    assert clean_name(raw) == ""
    assert clean_address(raw) == ""
    assert clean_country(raw) == ""
    assert parse_address(raw, "us") == {"address": "", "postal_code": "", "state": "", "city": "", "country": "us"}


def test_non_string_input_is_stringified():
    assert clean_name(123) == "123"


def test_nfkc_fullwidth():
    assert clean_name("ＡＢＣ Ｌｔｄ") == "abc ltd"


# ---------------------------------------------------------------- transliteration / accents

@pytest.mark.parametrize("raw, expected", [
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketimg pvt ltd"),      # Devanagari
    ("గుజరాత్ Logistics లిమిటెడ్", "gujrat logistics ltd"),            # Telugu
    ("आदित्य प्रॉपर्टीज एलएलपी", "adity proprtij llp"),                 # Devanagari LLP -> "elelpi"
])
def test_indic_scripts_transliterated(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("PLLC Novent Ówl", "pllc novent owl"),
    ("SJ ACE VENDOME ÍNC", "sj ace vendome inc"),
    ("Private Ambernath Sólar Limited", "pvt ambernath solar ltd"),
    ("LLC Moncada Léarning Center", "llc moncada learning center"),
    ("Façade Élève Crème", "facade eleve creme"),  # French accents (constructed)
])
def test_accents_stripped(raw, expected):
    assert clean_name(raw) == expected


def test_native_script_state_in_address():
    assert parse_address("74/1 EAST MOTI BAGH, SARAI ROHILLA, DELHI, दिल्ली", "india")["state"] == "dl"
    assert parse_address("PLOT NO. ##74, BHOPAL, मध्य प्रदेश", "india")["state"] == "mp"


# ---------------------------------------------------------------- names

@pytest.mark.parametrize("raw, expected", [
    ("Orelee's Barbershop", "orelee s barbershop"),
    ("Teodora  Pough Landscaping Inc", "teodora pough landscaping inc"),
    ("-- Holloway Peak Inc Seafood", "holloway peak inc seafood"),
    ("Novent [Owl]", "novent owl"),
    ("M/s Haven (Edm)", "haven edm"),
])
def test_name_punctuation_and_spaces(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Nova & Co", "nova and co"),
    ("B+ Retail Inc", "b and retail inc"),
])
def test_ampersand(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Tirupati Healthcare Private Limited", "tirupati healthcare pvt ltd"),
    ("Tirupati Healthcare Pvt. Ltd.", "tirupati healthcare pvt ltd"),
    ("Acme Corporation", "acme corp"),
    ("Acme Corp.", "acme corp"),
    ("Acme Incorporated", "acme inc"),
    ("Acme Inc.", "acme inc"),
    ("Aadarsh Exports Company", "aadarsh exports co"),
    ("Aadarsh Exports Co", "aadarsh exports co"),
    ("Smith L.L.C.", "smith llc"),
    ("Jones P.C.", "jones pc"),
    ("राम मार्केटिंग प्रा. लि.", "ram marketimg pvt ltd"),  # Hindi abbreviation
])
def test_legal_forms_kept_short(raw, expected):
    assert clean_name(raw) == expected


def test_legal_forms_consistent_across_spellings():
    variants = ["Goyal Impex Private Limited", "GOYAL IMPEX PVT LTD", "Goyal Impex Pvt. Ltd.",
                "गोयल इम्पेक्स प्राइवेट लिमिटेड"]
    cleaned = {clean_name(v) for v in variants[:3]}
    assert cleaned == {"goyal impex pvt ltd"}
    assert clean_name(variants[3]).endswith("pvt ltd")


@pytest.mark.parametrize("raw, expected", [
    ("sjacevendome.com", "sjacevendome"),
    ("AVISSOLUTIONS.COM", "avissolutions"),
    ("www.wilfordhancock.com", "wilfordhancock"),
    ("#sjace", "sjace"),
])
def test_website_and_hashtag_names(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("M/s Haven Exim", "haven exim"),
    ("M/S Zetagild Traders", "zetagild traders"),
    ("M/s. Zetagild Traders", "zetagild traders"),
])
def test_ms_prefix_dropped(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Haven Exim Exim", "haven exim"),
    ("Haven Haven (Services)", "haven services"),
    ("Goyal Limited Ltd", "goyal ltd"),  # collapses after canonicalization
])
def test_repeated_words_collapsed(raw, expected):
    assert clean_name(raw) == expected


def test_name_st_is_saint():
    assert clean_name("St Jude Clinic") == "saint jude clinic"


# ---------------------------------------------------------------- aliases (fka / dba / aka)

@pytest.mark.parametrize("raw, expected", [
    ("Zetagild fka Yukthi Hardware Private Limited", "yukthi hardware pvt ltd"),
    ("Zetaxylo F/K/A Doyle & Malik Information LLC", "doyle and malik information llc"),
    ("Wexjax formerly Horizon Mountain Securities LP", "horizon mountain securities lp"),
    ("Drexnoviio Formerly Harrison Tailwind", "harrison tailwind"),
    ("Onyxquo formerly known as Vaia Inc", "vaia inc"),
    ("Korjaxio a/k/a Crochet, Palmer and Flett LP", "crochet palmer and flett lp"),
    ("Xylobrix aka Midwest Disciplined Inc", "midwest disciplined inc"),
    ("Beloavi d/b/a Novent Owl PLLC", "novent owl pllc"),
    ("Quonex dba M D Herrera Offshore", "m d herrera offshore"),
    ("Quonex DBA M D Herrera Offshore", "m d herrera offshore"),
    ("Belocalo Co FKA Deleon's Select Pizza", "deleon s select pizza"),
])
def test_alias_keeps_part_after_marker(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Media Aka Services", "media aka services"),
    ("Aka Solution Ltd", "aka solution ltd"),
    ("LLP AKA MEDICAL", "llp aka medical"),
    ("Unique Dba Pvt. Ltd.", "unique dba pvt ltd"),
    ("DBA CONSTRUCTION PVT LTD", "dba construction pvt ltd"),  # marker at start: no split
    ("M/s DBA CONSTRUCTION PVT LTD", "dba construction pvt ltd"),
    ("Zetagild fka", "zetagild fka"),                           # nothing after: no split
    ("formerly Harrison Tailwind", "formerly harrison tailwind"),
])
def test_alias_not_split(raw, expected):
    assert clean_name(raw) == expected


def test_alias_markers_are_exact_spellings():
    for marker in ["aka", "a/k/a", "dba", "DBA", "d/b/a", "fka", "FKA", "f/k/a", "F/K/A",
                   "formerly", "Formerly", "formerly known as"]:
        assert resolve_alias(f"X {marker} Y") == " Y", marker
    for not_marker in ["Aka", "AKA", "Dba", "dBa", "Fka", "FORMERLY", "akas", "xaka"]:
        assert resolve_alias(f"X {not_marker} Y") == f"X {not_marker} Y", not_marker


# ---------------------------------------------------------------- address: abbreviations

@pytest.mark.parametrize("raw, expected", [
    ("105 ELM ST, MORGANTON, NC", "105 elm street, morganton, nc"),
    ("9327 ATWOOD DR, ST LOUIS, MO", "9327 atwood drive, saint louis, mo"),
    ("118 Main Street, St Charles, MO", "118 main street, saint charles, mo"),
    ("5600 S 3rd St, Louisville, KY", "5600 s 3rd street, louisville, ky"),
    ("12 Elm St NW, Olympia, WA", "12 elm street nw, olympia, wa"),
    ("17560 Ellis Rd, Tahlequah, OK", "17560 ellis road, tahlequah, ok"),
    ("1 Park Ave, New York, NY", "1 park avenue, new york, ny"),
    ("G Blk, Sector 5, Noida, UP", "g block, sector 5, noida, up"),
    ("12, Dr G D Marg, Mumbai, Maharashtra", "12, dr g d marg, mumbai, mh"),
])
def test_address_abbreviations(raw, expected):
    assert clean_address(raw) == expected


def test_address_french_accents():
    assert clean_address("12 Rue de la Paix Élysée, Besançon") == "12 rue de la paix elysee, besancon"


def test_address_junk_removed():
    assert clean_address("##16978 Moore Rd, <NULL>, Andalusia, Alabama") == "16978 moore road, andalusia, al"
    assert clean_address("003808 Holman St, Houston, Texas") == "3808 holman street, houston, tx"
    assert clean_address("A/##404, PRATHNA GREENS") == "a 404, prathna greens"


def test_address_keeps_numbers_with_slash_and_dash():
    assert clean_address("D.no.06-7-36/b/z, 7/1, Arundelpet") == "d no 06-7-36 b z, 7/1, arundelpet"


# ---------------------------------------------------------------- address: Opp (3a)

def test_opp_city_stays_opp():
    assert clean_address("116 Southern Oaks Drive, Unit Apartment 704, Opp, AL") == \
        "116 southern oaks drive, apartment 704, opp, al"
    assert parse_address("OPP, AL, 1041 483", "us")["address"] == "opp, al, 1041 483"


@pytest.mark.parametrize("raw, expected", [
    ("H.NO 008 GOVIND BUILDING OPP.HOTEL VRINDHAVAN", "h no 8 govind building opp hotel vrindhavan"),
    ("Opposite Royal Hotel, Juhu", "opp royal hotel, juhu"),
    ("Opp. Bank Of Baroda, Bandra", "opp bank of baroda, bandra"),
    ("301/306, Kunj Plaza, Opp, Fortune Gateway", "301/306, kunj plaza, opp, fortune gateway"),
])
def test_opp_landmark_kept_short(raw, expected):
    assert clean_address(raw) == expected


def test_landmark_phrase_intact():
    assert clean_address("Nr New Post Office, Mumbai") == "near new post office, mumbai"
    assert clean_address("Near SBI ATM, Hubli") == "near sbi atm, hubli"


# ---------------------------------------------------------------- address: West/East (3c)

@pytest.mark.parametrize("raw, expected", [
    ("Bandra (We, St), Mumbai, Maharashtra", "bandra west, mumbai, mh"),
    ("Santacruz (Ea, St), Mumbai, Maharashtra", "santacruz east, mumbai, mh"),
    ("KANDIVALI (WE, ST), MUMBAI", "kandivali west, mumbai"),
    ("Andheri (W), Mumbai", "andheri west, mumbai"),
    ("Andheri (West), Mumbai", "andheri west, mumbai"),
    ("Goregaon (E), Mumbai", "goregaon east, mumbai"),
    ("Goregaon (East), Mumbai", "goregaon east, mumbai"),
])
def test_west_east_suffix(raw, expected):
    assert clean_address(raw) == expected


def test_west_east_forms_agree():
    forms = ["Andheri (Ea, St), Mumbai", "Andheri (E), Mumbai", "Andheri (East), Mumbai"]
    assert {clean_address(f) for f in forms} == {"andheri east, mumbai"}


def test_west_east_orphans():
    # Reordered "(Ea, St)" split across parts (real S2/S3 records).
    assert clean_address("M-3, BALARAMA BUILDING, BANDRA (EA, MUMBAI, ST), महाराष्ट्र") == \
        "m 3, balarama building, bandra east, mumbai, mh"
    assert clean_address("Shop No.2, St), Mumbai, MH") == "shop no 2, mumbai, mh"


def test_po_st_no_special_handling():
    # Not rewritten; the generic "st)" orphan rule drops its second half.
    assert clean_address("Kalyan (Po, St), Thane") == "kalyan po, thane"


def test_dash_kept_only_between_digits():
    # "M-3" and "M-#1" (same building, different sources) both lose the dash.
    assert clean_address("M-3") == "m 3"
    assert clean_address("M-#1") == clean_address("M-1") == "m 1"
    assert clean_address("54-18-45") == "54-18-45"


# ---------------------------------------------------------------- address: Fl 0 (3d)

@pytest.mark.parametrize("raw, expected", [
    ("1644 Crownsville Road, Fl 0, Crownsville, MD", "1644 crownsville road, crownsville, md"),
    ("##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri", "8 willow oak lane, saint louis, mo"),
    ("507-B Pecos St, Fl. 0, Eagle Pass, Texas", "507 b pecos street, eagle pass, tx"),
    ("20 Main St, Fl 3, Austin, TX", "20 main street, floor 3, austin, tx"),
    ("20 Main St, Fl 12, Austin, TX", "20 main street, floor 12, austin, tx"),
    ("6 Flr, 603, B & C Wing", "6 floor, 603, b c wing"),
])
def test_floor_zero_dropped_real_floors_kept(raw, expected):
    assert clean_address(raw) == expected


# ---------------------------------------------------------------- address: components (3b, D4)

def test_washington_city_vs_state():
    p = parse_address("1 Main St, Washington, DC", "us")
    assert (p["address"], p["city"], p["state"]) == ("1 main street, washington, dc", "washington", "dc")
    assert parse_address("100 Pine St, Seattle, Washington", "us")["state"] == "wa"


def test_state_only_at_last_part_or_next_to_country():
    assert parse_address("Kolkata, West Bengal, Howrah, 57/3G", "india")["state"] == ""
    p = parse_address("12 MG Road, Pune, Maharashtra, India", "india")
    assert (p["state"], p["city"]) == ("mh", "pune")


def test_state_code_part_not_expanded():
    # Found by hypothesis: "Connecticut" -> "ct", which must not become "court" on re-cleaning.
    assert clean_address("12 Elm Ct, Hartford, CT") == "12 elm court, hartford, ct"
    assert clean_address("1 Ocean Dr, Miami, FL") == "1 ocean drive, miami, fl"
    assert clean_address(clean_address("CONNECTICUT")) == "ct"


def test_kansas_city_not_a_state():
    p = parse_address("1 Main St, Kansas City, MO", "us")
    assert (p["city"], p["state"]) == ("kansas city", "mo")


@pytest.mark.parametrize("raw, city, state", [
    ("9327 ATWOOD DR, ST LOUIS, MO", "saint louis", "mo"),
    ("932 Atwood Dr, Fl 0, Stlouis, Missouri", "stlouis", "mo"),
    ("34B Lenin Sarani, Kolkata, Calcutta, West Bengal", "kolkata", "wb"),
    ("H.no 4-0 Flat No. 103, Sai Bhavana Encl Kamala Nagar, Hyderabad, TG", "hyderabad", "tg"),
    ("PLOT NO B-78/1, AMBERNATH EAST, THANE, महाराष्ट्र", "thane", "mh"),
    ("A/404, Gandhinagar, Gandhi Nagar, GJ", "gandhi nagar", "gj"),
])
def test_city_and_state(raw, city, state):
    p = parse_address(raw, "us" if state in {"mo"} else "india")
    assert (p["city"], p["state"]) == (city, state)


def test_repeated_parts_removed():
    assert clean_address("Kolkata, Kolkata, Howrah, West Bengal") == "kolkata, howrah, wb"
    assert clean_address("Mumbai, Mumbai City, Maharashtra") == "mumbai, mh"


def test_city_aliases_whole_part_only():
    assert clean_address("Andheri, Bombay") == "andheri, mumbai"
    assert clean_address("34B Lenin Sarani, Calcutta") == "34b lenin sarani, kolkata"
    assert clean_address("Opp. Bank Of Baroda, Bandra") == "opp bank of baroda, bandra"


@pytest.mark.parametrize("raw, country, postal", [
    ("12 MG Road, Pune, Maharashtra 411001", "india", "411001"),
    ("12 MG Road, Pune, 411001", "india", "411001"),
    ("12 MG Road, Pin: 411001, Pune", "india", "411001"),
    ("12 MG Road, Pincode - 411001, Pune", "india", "411001"),
    ("1 Main St, Austin, TX 78701", "us", "78701"),
    ("1 Main St, Austin, TX 78701-1234", "us", "78701-1234"),
    ("1 Main St, Austin, TX, 78701", "us", "78701"),
])
def test_postal_code_unmistakable(raw, country, postal):
    assert parse_address(raw, country)["postal_code"] == postal


@pytest.mark.parametrize("raw, country", [
    ("17560 Ellis Road, Tahlequah, OK", "us"),                 # 5-digit house number at start
    ("Saint Joseph, Indiana, 002412 Hobson Road", "us"),       # zero-padded house number
    ("NO 00925 APARTMENT D-26, BANGALORE, Karnataka", "india"),
    ("B-00206, Big Splash, Sector 17 Vashi, Thane, MH", "india"),
    ("Surat, 213, Surat, Gujarat, 008006", "india"),           # PIN never starts with 0
    ("12 Rue de Rivoli, 75001 Paris", "france"),               # no postal rule for other countries
    ("12 Main St, Austin, TX 78701", "india"),                 # US shape under another country
])
def test_postal_code_not_guessed(raw, country):
    assert parse_address(raw, country)["postal_code"] == ""


def test_full_cleaned_address_kept_alongside_components():
    p = parse_address("Des Plaines, IL, 9308 Home Court", "us")
    assert p["address"] == "des plaines, il, 9308 home court"
    assert p["country"] == "us"


# ---------------------------------------------------------------- country (point 5)

@pytest.mark.parametrize("raw, expected", [
    ("US", "us"), ("India", "india"), ("France", "france"), ("  France ", "france"),
    ("Côte d'Ivoire", "cote d ivoire"), ("United  States", "united states"),
])
def test_clean_country_open_set(raw, expected):
    assert clean_country(raw) == expected


@pytest.mark.parametrize("variant", ["US", "us", " Us ", "US\t"])
def test_country_normalized_once_at_entry(variant):
    rec = {"entity_id": "S2-1", "business_name": "Acme Inc",
           "business_address": "1 Main St, Austin, TX 78701", "country": variant}
    out = clean_record(rec)
    assert out["country_clean"] == "us"
    assert out["postal_code"] == "78701"


def test_downstream_receives_clean_country(monkeypatch):
    seen = []
    real = clean.parse_address
    monkeypatch.setattr(clean, "parse_address", lambda a, c: seen.append(c) or real(a, c))
    clean_record({"business_name": "X", "business_address": "Y", "country": "  FRANCE "})
    assert seen == ["france"]
    assert type(seen[0]) is str


def test_no_lookup_uses_raw_country():
    """Every country-keyed table is keyed by clean (lowercase) country, and a raw
    country passed downstream by mistake matches nothing."""
    for table in (POSTAL_AT_END, POSTAL_AFTER_PIN):
        for key in table:
            assert key == clean_country(key), key
    assert parse_address("1 Main St, Austin, TX 78701", "US")["postal_code"] == ""
    assert parse_address("1 Main St, Austin, TX 78701", "us")["postal_code"] == "78701"


def test_clean_record_never_drops_fields():
    rec = {"entity_id": "S1-1", "business_name": None, "business_address": None, "country": None}
    out = clean_record(rec)
    assert {k: out[k] for k in rec} == rec
    assert out["name_clean"] == out["address_clean"] == out["country_clean"] == ""


# ---------------------------------------------------------------- determinism

def test_deterministic_repeated_calls():
    raw = "Zetagild fka Yukthi Hardware प्राइवेट Limited", "Bandra (We, St), Opp, Mumbai, महाराष्ट्र"
    first = (clean_name(raw[0]), parse_address(raw[1], "india"))
    for _ in range(5):
        assert (clean_name(raw[0]), parse_address(raw[1], "india")) == first


# ---------------------------------------------------------------- property tests

any_text = st.one_of(st.none(), st.just(math.nan), st.text(), st.text(alphabet=st.characters(codec="utf-8")))
# Text built from the tokens the rules care about, so hypothesis exercises them.
RULE_WORDS = ["st", "ste", "dr", "fl", "0", "00", "opp", "(we, st)", "(ea", "st)", "(w)", "(e)", "city",
              "unit", "kansas", "illinois", "washington", "dc", "india", "us", "m/s", "pvt", "limited",
              "pra", "li", ".com", "l.l.c.", "&", "fka", "dba", "DBA", "Aka", "<NULL>", "nan", "##", "-", "/",
              "maharashtra", "महाराष्ट्र", "é", "Ó"]
rule_text = st.lists(st.sampled_from(RULE_WORDS + [" ", ", ", ","]), max_size=12).map("".join)
texts = st.one_of(any_text, rule_text)
countries = st.sampled_from(["", "us", "india", "france"])


@settings(max_examples=500)
@given(texts)
def test_never_raises(raw):
    clean_name(raw)
    clean_address(raw)
    clean_country(raw)
    clean_record({"business_name": raw, "business_address": raw, "country": raw})


@settings(max_examples=500)
@given(texts, countries)
def test_output_ascii(raw, country):
    for out in (clean_name(raw), clean_country(raw), *parse_address(raw, country).values()):
        assert out.isascii()


@settings(max_examples=500)
@given(texts)
def test_name_idempotent(raw):
    once = clean_name(raw)
    # Alias markers are case-sensitive on raw input; once lowercased, a real word like
    # "Media Aka Services" -> "media aka services" looks like a marker. Documented limit.
    assume(not ALIAS_MARKER.search(once))
    assert clean_name(once) == once


@settings(max_examples=500)
@given(texts, countries)
def test_address_idempotent(raw, country):
    once = clean_address(raw, country)
    assert clean_address(once, country) == once


@settings(max_examples=500)
@given(texts)
def test_country_idempotent(raw):
    once = clean_country(raw)
    assert clean_country(once) == once
