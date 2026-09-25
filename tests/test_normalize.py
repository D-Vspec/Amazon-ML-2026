import pandas as pd
import pytest

from er.normalize import INDIA_STATES, LEGAL_FORMS, US_STATES, RuleNormalizer
from er.transliterate import AnyAsciiTransliterator

n = RuleNormalizer()
t = AnyAsciiTransliterator()


def name(s):
    return n.normalize_name(s)[0]


def legal(s):
    return n.normalize_name(s)[1]


# ---------------------------------------------------------------- names: core

@pytest.mark.parametrize("raw, expected", [
    ("Orelee's Barbershop", "orelee s barbershop"),
    ("HARRIS BETTER", "harris better"),
    ("Harris  Better", "harris better"),                       # double space
    ("  Prime Money  ", "prime money"),                        # outer whitespace
    ("Precision-Telecom  Dynamics, Inc", "precision telecom dynamics"),
    ("Shelbi Robinson [Graf]", "shelbi robinson graf"),        # brackets
    ("PATRIOT  (ASSOCIATION)", "patriot association"),         # parens
    ("*** PATRIOT ASSOCIATION", "patriot association"),        # junk prefix
    ("-- Holloway Peak Inc Seafood", "holloway peak seafood"),
    ("<< Team Ecole", "team ecole"),
    ('"Quoted" Name', "quoted name"),
    ("J 6 Precision Axiom", "j 6 precision axiom"),            # digits kept
    ("B+ Retail Inc", "b and retail"),                         # + -> and
    ("Kellen, Kempf & Richard Environmental LLC", "kellen kempf and richard environmental"),
    ("Sanchez and Bryant Corp", "sanchez and bryant"),
])
def test_name_core(raw, expected):
    assert name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Federati0n", "federation"),
    ("Fe1iciano Universal", "feliciano universal"),
    ("Envir0nmental", "environmental"),
    ("J 6 Precision", "j 6 precision"),      # digit between spaces stays
    ("Route 66 Diner", "route 66 diner"),     # real numbers stay
    ("3M", "3m"),
    ("A1 Bakery", "a1 bakery"),               # digit not between two letters
])
def test_name_leetspeak(raw, expected):
    assert name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("gajananservices.com", "gajananservices"),
    ("Piedmontnetwork.Com", "piedmontnetwork"),
    ("*** Digitalprivatepranya.Com", "digitalprivatepranya"),
    ("www.example.com", "example"),
    ("rayaindiadesigns.in", "rayaindiadesigns"),
    ("shop.co.in", "shop"),
    ("boutique.fr", "boutique"),
    ("charity.org", "charity"),
])
def test_name_websites(raw, expected):
    assert name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Fayewexhalo DBA Precision Telecom Dynamics, Inc", "precision telecom dynamics"),
    ("Orbilumecto trading as Andromeda Seva Samiti", "andromeda seva samiti"),
    ("Acme d/b/a Road Runner Supply", "road runner supply"),
    ("Acme t/a Road Runner Supply", "road runner supply"),
    ("Adbac Holdings", "adbac holdings"),     # "dba" inside a word is not a DBA marker
])
def test_name_dba(raw, expected):
    assert name(raw) == expected


# ---------------------------------------------------------------- names: legal form

@pytest.mark.parametrize("raw, core, form", [
    ("Meighan Wheeler P.C.", "meighan wheeler", "pc"),
    ("Meighan Wheeler Wheeler (P.C.)", "meighan wheeler wheeler", "pc"),
    ("Meighan P.C. Wheeler", "meighan wheeler", "pc"),                 # suffix moved inside
    ("L.L.C. Clyial Pony", "clyial pony", "llc"),                      # prefix
    ("LLC CLYIAL PONY", "clyial pony", "llc"),
    ("Clyial Pony-LLC", "clyial pony", "llc"),
    ("Cypress Diana Inc Inc", "cypress diana", "inc"),                 # duplicate
    ("Agro India Private Limited", "agro india", "ltd pvt"),
    ("AGRO INDIA PRIVATE  LTD", "agro india", "ltd pvt"),
    ("Gajanan Limited Private (Services)", "gajanan services", "ltd pvt"),
    ("Smt Gajanan Services Limited-Private", "smt gajanan services", "ltd pvt"),
    ("Bond Tecealom Pvt Pvt Ltd", "bond tecealom", "ltd pvt"),
    ("Bond Telecom Pvt  (Ltd)", "bond telecom", "ltd pvt"),
    ("Design Builders Company", "design builders", "co"),
    ("Sanchez and Bryant (Corp.)", "sanchez and bryant", "corp"),
    ("Acme Corporation", "acme", "corp"),
    ("Acme Incorporated", "acme", "inc"),
    ("Southern Builders LLP", "southern builders", "llp"),
    ("Acme PLLC", "acme", "pllc"),
    ("Acme LP", "acme", "lp"),
    ("Acme OPC Private Limited", "acme", "ltd opc pvt"),
    ("SCI Ptit Amicale", "ptit amicale", "sci"),                       # French forms
    ("Boulangerie Martin SARL", "boulangerie martin", "sarl"),
    ("Martin SAS", "martin", "sas"),
    ("Martin SASU", "martin", "sasu"),
    ("Martin EURL", "martin", "eurl"),
    ("Prime Money", "prime money", ""),                                # no legal form
])
def test_legal_form_split(raw, core, form):
    assert n.normalize_name(raw) == (core, form)


@pytest.mark.parametrize("raw, core", [
    ("Ss & Co", "ss"),                   # dangling "and" removed
    ("*** Ss Co  &", "ss"),
    ("Cotton & Co Co", "cotton"),
    ("(CO) COTTON & CO", "cotton"),
    ("& Sons Ltd", "sons"),              # leading "and" removed
    ("Sri Ss + Company", "sri ss"),
])
def test_dangling_and(raw, core):
    assert name(raw) == core


@pytest.mark.parametrize("raw, core", [
    ("Inc", "inc"),
    ("Private Limited", "pvt ltd"),
    ("LLC", "llc"),
    ("Co.", "co"),
])
def test_name_made_only_of_legal_words_keeps_them(raw, core):
    assert name(raw) == core


@pytest.mark.parametrize("raw", [
    "Costco Wholesale", "Incline Partners", "Limitless Media", "Privateer Yachts",
    "Colt Arms", "Cooper Tires", "Corpus Christi Bakery", "Sasha Design", "Llcoolj Music",
    "Pcs Technologies", "Scientific Labs", "Coinbase",
])
def test_legal_words_only_matched_as_whole_tokens(raw):
    assert legal(raw) == ""
    assert name(raw) == raw.lower()


# Transliterated "private limited" from every script in the data.
@pytest.mark.parametrize("raw", [
    "praivet limited", "piraivet limitet", "praibhet limited", "praivrr limirrd",
    "pra. li.", "pra li", "Pvt. Ltd.", "Private Limited", "PVT LTD",
])
def test_pvt_ltd_variants_share_legal_form(raw):
    assert legal("acme " + raw) == "ltd pvt"
    assert name("acme " + raw) == "acme"


def test_llp_spelled_out():
    assert n.normalize_name("adity proprtij elelpi") == ("adity proprtij", "llp")


def test_pra_li_needs_both_parts():
    assert name("Pra Holdings") == "pra holdings"
    assert name("Li Wei Trading") == "li wei trading"


@pytest.mark.parametrize("raw", LEGAL_FORMS)
def test_every_legal_form_is_extracted(raw):
    assert legal(f"acme {raw}") == LEGAL_FORMS[raw]


def test_legal_form_sorted_and_unique():
    assert legal("Acme Pvt Ltd Pvt Limited Private") == "ltd pvt"


# ---------------------------------------------------------------- names: variants of one business agree

@pytest.mark.parametrize("variants", [
    ["Precision Telecom Dynamics, Inc", "Precision-Telecom  Dynamics, Inc",
     "Precision Telecom Dynamics, INC", "Fayewexhalo DBA Precision Telecom Dynamics, Inc"],
    ["Agro India Private Limited", "AGRO INDIA PRIVATE  LTD", "Agro India Private  Limited"],
    ["Clyial Pony LLC", "L.L.C. Clyial Pony", "CLYIAL PONY LLC", "LLC CLYIAL PONY", "Clyial Pony-LLC"],
    ["Empire Federation", "Empire  Federation", "EMPIRE FEDERATION", "Empire Federati0n"],
    ["Kellen, Kempf & Richard Environmental LLC", "Kellen, Kempf & Richard Envir0nmental LLC"],
    ["Sanchez and Bryant Corp", "Sanchez and Bryant Corp.", "Sanchez and Bryant (Corp.)", "Sanchez & Bryant"],
    ["Feliciano Universal Group", "Fe1iciano-Universal Group", "FELICIANO UNIVERSAL GROUP",
     "Feliciano Universal-Group"],
    ["Patriot Association", "patriot association", "*** PATRIOT ASSOCIATION", "PATRIOT  (ASSOCIATION)"],
])
def test_variants_normalize_to_same_name(variants):
    assert len({name(v) for v in variants}) == 1


@pytest.mark.parametrize("a, b", [
    ("Ram Marketing", "Ram Traders"),
    ("Precision Telecom", "Precision Telecom Dynamics"),
    ("Harris Better", "Harris Bitter"),
    ("South It Limited", "North It Limited"),
])
def test_different_businesses_stay_different(a, b):
    assert name(a) != name(b)


# ---------------------------------------------------------------- names: after transliteration

@pytest.mark.parametrize("native, core, form", [
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketimg", "ltd pvt"),
    ("आदित्य प्रॉपर्टीज एलएलपी", "adity proprtij", "llp"),
    ("राम मार्केटिंग प्रा. लि.", "ram marketimg", "ltd pvt"),
    ("ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "guru estet", "ltd pvt"),
    ("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "kulopl picins", "ltd pvt"),
    ("ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്", "skti impeks", "ltd pvt"),
    ("ইনোভেটিভ প্রোডাক্টস রেস্টুরেন্ট লিমিটেড", "inobhetibh prodakts resturent", "ltd"),
    ("Red मीडिया प्राइवेट लिमिटेड", "red midiya", "ltd pvt"),
    ("SCI Ptit Àmicale", "ptit amicale", "sci"),
])
def test_after_transliteration(native, core, form):
    assert n.normalize_name(t.transliterate(native)) == (core, form)


# ---------------------------------------------------------------- names: edge cases

@pytest.mark.parametrize("raw", ["", " ", "***", "&", "( )", "--"])
def test_empty_or_junk_name(raw):
    assert n.normalize_name(raw) == ("", "")


@pytest.mark.parametrize("raw", [
    "Meighan Wheeler P.C.", "Fe1iciano-Universal Group", "*** Ss Co  &", "gajananservices.com",
    "Kellen, Kempf & Richard Environmental LLC", "Agro India Private Limited", "B+ Retail Inc",
])
def test_name_core_is_idempotent(raw):
    once = name(raw)
    assert name(once) == once


# ---------------------------------------------------------------- addresses

@pytest.mark.parametrize("raw, expected", [
    ("1795 Westchester Drive, High Point, NC", "1795 westchester drive, high point, nc"),
    ("105 ELM ST, MORGANTON, NC", "105 elm street, morganton, nc"),
    ("003160 SWAINSONS LN, VIRGINIA BEACH, VA", "3160 swainsons lane, virginia beach, va"),
    ("00109 JACKSON ST, SCOTT CITY, KS", "109 jackson street, scott, ks"),
    ("409 MARLBOROUGH ST, <NULL>, BOSTON, MA", "409 marlborough street, boston, ma"),
    ("0409 Marlborough St, Unit 42, Boston, Massachusetts", "409 marlborough street, 42, boston, ma"),
    ("0409 Marlborough Street, # 42, Boston, Massachusetts", "409 marlborough street, 42, boston, ma"),
    ("#3343 Dug Hill Rd, Huntsville CITY, Alabama", "3343 dug hill road, huntsville, al"),
    ("1902 VISTA RIDGE CT, null, DRAPER CITY (UTAH CO), UT", "1902 vista ridge court, draper utah co, ut"),
    ("Ohio, # E, 1232 Channingway Dr, Fairborn", "oh, e, 1232 channingway drive, fairborn"),
    ("60-23 69 Lane, Maspeth, New York", "60-23 69 lane, maspeth, ny"),
    ("4911 Kessler Ave, Fl 0, Wichita, Kansas", "4911 kessler avenue, floor 0, wichita, ks"),
    ("2159 Betsy'S Way, Kaysville, Utah", "2159 betsy s way, kaysville, ut"),
    ("10-12 Paradise Cir, Caddo Valley, Arkansas", "10-12 paradise circle, caddo valley, ar"),
    ("19204 112TH STREET, BONNEY LAKE, WA", "19204 112th street, bonney lake, wa"),
])
def test_us_addresses(raw, expected):
    assert n.normalize_address(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "kh no 570/13, new delhi, west delhi, dl"),
    ("# #163-F , HSIIDC, SECTOR-3, KARNAL, hriyana", "163 f, hsiidc, sector 3, karnal, hr"),
    ("Flat No. B-##901, Oakwood Hills, S.no.26/34, Opp. Pan Card C Lub, Baner, Pune, MH",
     "flat no b 901, oakwood hills, s no 26/34, opposite pan card c lub, baner, pune, mh"),
    ("5-4-18-45, Ff-501, 4Th Flr, Taruni Vista Apts", "5-4-18-45, ff 501, 4th floor, taruni vista apartments"),
    ("HOUSE NO - 007, BLOCK - A SECTOR -19, DWARKA, NEW DELHI, dilli",
     "house no 7, block a sector 19, dwarka, new delhi, dl"),
    ("Flat - 2D., N/A, Kolkata, Howrah, pscimbng", "flat 2d, kolkata, howrah, wb"),
    ("PLOT N.B3/275&276, GALI N.39", "plot n b3/275 276, gali n 39"),
    ("W 232 D, PHASE II, MIDC, DOMBIVLI EAST, KALYAN-THANE, DOMBIVLI, Maharashtra",
     "w 232 d, phase ii, midc, dombivli east, kalyan thane, dombivli, mh"),
    ("B3/394-395 Somasundaram Mill Road Annuparpalayam, Coimbatore, TN",
     "b3/394-395 somasundaram mill road annuparpalayam, coimbatore, tn"),
    ("Magadi Village, Hassan District, krnatk", "magadi village, hassan district, ka"),
    ("S/O SHRIPATI KARANJE AT POST.SULTANPUR BK TQ.SHEVGAON DIST.AHMEDNAGAR",
     "s o shripati karanje at post sultanpur bk tq shevgaon district ahmednagar"),
])
def test_india_addresses(raw, expected):
    assert n.normalize_address(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("12 Rue de la Paix, Paris", "12 rue de la paix, paris"),
    ("5 Av Victor Hugo, Lyon", "5 avenue victor hugo, lyon"),
    ("8 Bd Haussmann, Paris", "8 boulevard haussmann, paris"),
])
def test_france_addresses_use_generic_rules(raw, expected):
    assert n.normalize_address(raw) == expected


@pytest.mark.parametrize("abbr, full", [
    ("st", "street"), ("rd", "road"), ("ave", "avenue"), ("dr", "drive"), ("ln", "lane"),
    ("ct", "court"), ("pl", "place"), ("cir", "circle"), ("blvd", "boulevard"), ("hwy", "highway"),
    ("pkwy", "parkway"), ("ter", "terrace"), ("trl", "trail"), ("sq", "square"), ("flr", "floor"),
    ("bldg", "building"), ("opp", "opposite"), ("nr", "near"), ("dist", "district"),
])
def test_street_abbreviations(abbr, full):
    assert n.normalize_address(f"1 Main {abbr.upper()}") == f"1 main {full}"
    assert n.normalize_address(f"1 Main {full}") == f"1 main {full}"


@pytest.mark.parametrize("raw, expected", [
    ("1 Westchester Drive", "1 westchester drive"),   # "st" inside a word untouched
    ("1 Stanford Road", "1 stanford road"),
    ("1 Drake Lane", "1 drake lane"),
    ("1 Avenue Road", "1 avenue road"),
])
def test_street_words_only_replaced_as_whole_tokens(raw, expected):
    assert n.normalize_address(raw) == expected


@pytest.mark.parametrize("state, code", list(US_STATES.items()))
def test_every_us_state(state, code):
    assert n.normalize_address(f"1 Main St, Town, {state.title()}") == f"1 main street, town, {code}"


@pytest.mark.parametrize("state, code", list(INDIA_STATES.items()))
def test_every_india_state(state, code):
    assert n.normalize_address(f"Plot 1, Town, {state}") == f"plot 1, town, {code}"


@pytest.mark.parametrize("native, code", [
    ("महाराष्ट्र", "mh"), ("दिल्ली", "dl"), ("उत्तर प्रदेश", "up"), ("ಕರ್ನಾಟಕ", "ka"),
    ("தமிழ்நாடு", "tn"), ("ગુજરાત", "gj"), ("পশ্চিমবঙ্গ", "wb"), ("తెలంగాణ", "tg"),
    ("हरियाणा", "hr"), ("राजस्थान", "rj"), ("കേരളം", "kl"), ("बिहार", "br"),
    ("मध्य प्रदेश", "mp"), ("ఆంధ్రప్రదేశ్", "ap"), ("ਪੰਜਾਬ", "pb"), ("ଓଡ଼ିଶା", "od"),
])
def test_native_script_states_after_transliteration(native, code):
    assert n.normalize_address(t.transliterate(f"Plot 1, {native}")) == f"plot 1, {code}"


@pytest.mark.parametrize("raw, expected", [
    ("Kansas City, KS", "kansas city, ks"),                  # city named after a state
    ("Oklahoma City, Oklahoma", "oklahoma city, ok"),
    ("123 New York Avenue, Brooklyn, NY", "123 new york avenue, brooklyn, ny"),
    ("Virginia Beach, VA", "virginia beach, va"),
    ("West Delhi, Delhi", "west delhi, dl"),
])
def test_state_only_matched_as_whole_component(raw, expected):
    assert n.normalize_address(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("", ""),
    (" ", ""),
    ("null", ""),
    ("<NULL>", ""),
    ("N/A", ""),
    ("1 Main St, null, NA, none, -, Town", "1 main street, town"),
    (",,, ,", ""),
    ("#", ""),
])
def test_null_and_empty_components(raw, expected):
    assert n.normalize_address(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("003160 Main St", "3160 main street"),
    ("008008 14 Ave", "8008 14 avenue"),
    ("000 Main St", "0 main street"),
    ("0409-12 Main St", "0409-12 main street"),   # only pure numbers lose leading zeros
    ("B-0901", "b 901"),
])
def test_leading_zeros(raw, expected):
    assert n.normalize_address(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("26/34", "26/34"),
    ("54-18-45", "54-18-45"),
    ("Sector-3", "sector 3"),
    ("163-F", "163 f"),
    ("Kalyan-Thane", "kalyan thane"),
    ("S/O Ram", "s o ram"),
    ("-570/13", "570/13"),
    ("12/ Main", "12 main"),
])
def test_slash_and_hyphen_only_kept_between_digits(raw, expected):
    assert n.normalize_address(raw) == expected


def test_unit_markers_dropped():
    assert n.normalize_address("1 Main St, Unit E") == n.normalize_address("1 Main St, # E") == "1 main street, e"
    assert n.normalize_address("1 Main St, Suite 5") == n.normalize_address("1 Main St, Ste 5") == "1 main street, 5"


@pytest.mark.parametrize("variants", [
    ["1600 Treewood Lane, Chesterfield County, VA", "1600 TREEWOOD LANE, CHESTERFIELD COUNTY, VA"],
    ["8008 14 Avenue, Brooklyn, NY", "8008 14 AVE, BROOKLYN, NY", "008008 14 Ave, Brooklyn, New York"],
    ["2159 BETSY'S WAY, KAYSVILLE, UT", "2159 Betsy'S Way, Kaysville, Utah"],
    ["3343 DUG HILL RD, HUNTSVILLE, AL", "#3343 Dug Hill Road, Huntsville CITY, Alabama"],
    ["409 MARLBOROUGH ST, <NULL>, BOSTON, MA", "0409 Marlborough Street, Boston, Massachusetts"],
    ["1453 Road 5705, Cleveland, TX", "1453 ROAD 5705, CLEVELAND, Texas"],
    ["Flat - 2D., Kolkata, Howrah, WB", "Flat - 2D., Kolkata, Howrah, West Bengal", "Flat - 2D., Kolkata, Howrah, pscimbng"],
])
def test_address_variants_agree(variants):
    assert len({n.normalize_address(v) for v in variants}) == 1


def test_component_order_preserved():
    assert n.normalize_address("TX, CLEVELAND, 1453 ROAD 5705") == "tx, cleveland, 1453 road 5705"


@pytest.mark.parametrize("raw", [
    "0409 Marlborough St, Unit 42, Boston, Massachusetts",
    "# #163-F , HSIIDC, SECTOR-3, KARNAL, hriyana",
    "5-4-18-45, Ff-501, 4Th Flr, Taruni Vista Apts",
    "1902 VISTA RIDGE CT, null, DRAPER CITY (UTAH CO), UT",
    "Kansas City, KS",
])
def test_address_is_idempotent(raw):
    once = n.normalize_address(raw)
    assert n.normalize_address(once) == once


# ---------------------------------------------------------------- transform

def _records():
    return pd.DataFrame({
        "entity_id": ["S1-1", "S2-2", "S3-3"],
        "business_name": ["Agro India Private Limited", "L.L.C. Clyial Pony", ""],
        "business_address": ["Office 203, Mumbai, Maharashtra", "", "12 Rue de la Paix, Paris"],
        "country": ["India", "US", "France"],
    }, index=[5, 6, 7])


def test_transform_adds_columns():
    out = n.transform(_records())
    assert out.name_norm.tolist() == ["agro india", "clyial pony", ""]
    assert out.legal_form.tolist() == ["ltd pvt", "llc", ""]
    assert out.address_norm.tolist() == ["office 203, mumbai, mh", "", "12 rue de la paix, paris"]


def test_transform_keeps_raw_columns_and_index():
    df = _records()
    out = n.transform(df)
    for col in df.columns:
        assert out[col].tolist() == df[col].tolist()
    assert list(out.index) == [5, 6, 7]


def test_transform_does_not_mutate_input():
    df = _records()
    n.transform(df)
    assert "name_norm" not in df.columns


def test_transform_empty_frame():
    out = n.transform(_records().iloc[:0])
    assert out.empty
    assert {"name_norm", "legal_form", "address_norm"} <= set(out.columns)


def test_pipeline_transliterate_then_normalize():
    df = pd.DataFrame({
        "entity_id": ["S2-1"], "business_name": ["राम मार्केटिंग प्राइवेट लिमिटेड"],
        "business_address": ["KH NO. -570/13, NEW DELHI, दिल्ली"], "country": ["India"],
    })
    out = n.transform(t.transform(df))
    assert out.iloc[0][["name_norm", "legal_form", "address_norm"]].tolist() == \
        ["ram marketimg", "ltd pvt", "kh no 570/13, new delhi, dl"]
