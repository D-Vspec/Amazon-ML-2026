import pandas as pd
import pytest

from er.normalize import FRANCE_REGIONS, INDIA_STATES, LEGAL_FORMS, STATE_CODES, US_STATES, RuleNormalizer
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
    ("Prime Money", "prime money", ""),                                # no legal form
])
def test_legal_form_split(raw, core, form):
    assert n.normalize_name(raw) == (core, form)


@pytest.mark.parametrize("raw, core, form", [
    ("Rizavantage One D.B.A. Golden Tavern", "golden tavern", ""),
    ("Evobrix D.B.A. Kandivali East Stone Private Limited", "kandivali east stone", "ltd pvt"),
    ("Wexavi Labs D.B.A. Certified Energy Group PLLC", "certified energy group", "pllc"),
    ("Acme Pvt.Ltd.", "acme", "ltd pvt"),                 # dot between words separates them
    ("Acme Co.Ltd", "acme", "co ltd"),
    ("Acme Inc.", "acme", "inc"),
    ("Acme L.L.C", "acme", "llc"),                        # no trailing dot
    ("J.R. Industries", "jr industries", ""),
    ("J.R.Industries", "jr industries", ""),              # initials glued to the next word
    ("U.S.A. Motors Corp.", "usa motors", "corp"),
    ("St. Mary's Clinic", "saint mary s clinic", ""),
    ("No.1 Bakery", "no 1 bakery", ""),
    ("A0.A", "a0 a", ""),                                 # dot stops the leetspeak fix
])
def test_dots(raw, core, form):
    assert n.normalize_name(raw) == (core, form)
    assert name(core) == core


# Real France rows from the test split.
@pytest.mark.parametrize("raw, core, form", [
    ("Saint-Nazaire Primaire SAS", "saint nazaire primaire", "sas"),
    ("St-Nazaire Primaire SAS", "saint nazaire primaire", "sas"),
    ("GVI Comite Sarl", "gvi comite", "sarl"),
    ("Ets Asso EURL", "etablissements asso", "eurl"),
    ("Etablissements Familles EURL", "etablissements familles", "eurl"),
    ("Departemental Maison (France) SA", "departemental maison france", "sa"),
    ("Montreal & Cie SARL", "montreal", "co sarl"),
    ("Montreal et Cie SARL", "montreal", "co sarl"),
    ("Choeur & Cie France SAS", "choeur and france", "co sas"),
    ("SCI Ptit Àmicale", "ptit amicale", "sci"),
    ("Martin SASU", "martin", "sasu"),
    ("Martin EI", "martin", "ei"),
    ("Martin SNC", "martin", "snc"),
    ("Cabinet Dupont SELARL", "cabinet dupont", "selarl"),
    ("Martin Scop", "martin", "scop"),
    ("Martin GIE", "martin", "gie"),
    ("Martin Compagnie", "martin", "co"),
    ("Parents et Amis", "parents and amis", ""),
    ("Lycee Du [Marie]", "lycee du marie", ""),
    ("consciencesection.com", "consciencesection", ""),
    ("SA", "sa", "sa"),                                       # only a legal word: kept as the name
])
def test_france_names(raw, core, form):
    out = n.normalize_name(t.transliterate(raw), "France")
    assert out == (core, form)
    assert n.normalize_name(core, "France")[0] == core


@pytest.mark.parametrize("country", ["US", "India", ""])
@pytest.mark.parametrize("raw, core", [
    ("SAS Institute", "sas institute"),
    ("SA Enterprises", "sa enterprises"),
    ("EI Solutions", "ei solutions"),
    ("Sci Tech Labs", "sci tech labs"),
    ("Rahul Et Al", "rahul et al"),
])
def test_french_rules_only_in_france(country, raw, core):
    assert n.normalize_name(raw, country) == (core, "")


@pytest.mark.parametrize("raw, core", [
    ("St. Xavier's School", "saint xavier s school"),
    ("St Jude Clinic", "saint jude clinic"),
    ("Saint Jude Clinic", "saint jude clinic"),
    ("Stjude Clinic", "stjude clinic"),
])
def test_st_is_saint_in_names(raw, core):
    assert name(raw) == core


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


@pytest.mark.parametrize("raw", ["Prali Foods", "Pra1i Foods", "Pra-Lite Foods"])
def test_pra_li_needs_separator(raw):
    assert legal(raw) == ""


@pytest.mark.parametrize("raw, core", [
    ("PRA:PRA:LI:LI", "pra pra li li"),         # found by hypothesis: removing the inner pair used to expose a new one
    ("Acme pra Pvt li", "acme"),                # pair only adjacent once "Pvt" is pulled out as a legal form
])
def test_pra_li_pairs_formed_by_removal(raw, core):
    assert n.normalize_name(raw) == (core, "ltd pvt")
    assert name(core) == core


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
])
def test_after_transliteration(native, core, form):
    assert n.normalize_name(t.transliterate(native)) == (core, form)


# ---------------------------------------------------------------- names: edge cases

@pytest.mark.parametrize("raw", ["", " ", "***", "&", "( )", "--", "DBA", "d/b/a", "trading as"])
def test_empty_or_junk_name(raw):
    assert n.normalize_name(raw) == ("", "")


def test_french_et_alone_is_empty():  # real test-split row
    assert n.normalize_name("ET", "France") == ("", "")


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
def test_france_generic_rules_without_country(raw, expected):
    assert n.normalize_address(raw) == expected


# Real France rows from the test split.
@pytest.mark.parametrize("raw, expected", [
    ("13 R. DU PLESSIS, ST-NAZAIRE, Loire-Atlantique", "13 rue du plessis, saint nazaire, pdl"),
    ("58 R. MARCEAU, TOURCOING, Nord", "58 rue marceau, tourcoing, hdf"),
    ("63 R DE LA POTENTE, Tourcoing, Hauts-de-France", "63 rue de la potente, tourcoing, hdf"),
    ("Ndeg25 R BEETHOVEN, DUNKERQUE", "no 25 rue beethoven, dunkerque"),
    ("N° 25 Rue Beethoven, Dunkerque", "no 25 rue beethoven, dunkerque"),
    ("29 Boulevard Du Haut Livrac, Pessac, Nouvelle-Aquitaine", "29 boulevard du haut livrac, pessac, naq"),
    ("22 PLACE DAUPHINE, MERIGNAC, Gironde", "22 place dauphine, merignac, naq"),
    ("Merignac, Nouvelle-Aquitaine, 7 Impasse Guynemer", "merignac, naq, 7 impasse guynemer"),
    ("7 Imp. Guynemer, Merignac", "7 impasse guynemer, merignac"),
    ("NANTES, 5 AV DE LUSANSAY, Pays de la Loire", "nantes, 5 avenue de lusansay, pdl"),
    ("# 88 ROUTE DE FORT-MARDYCK, DUNKERQUE, Nord", "88 route de fort mardyck, dunkerque, hdf"),
    ("88 Rte de Fort-Mardyck, Dunkerque", "88 route de fort mardyck, dunkerque"),
    ("7 Allee Des Trois Lavoirs, Pessac", "7 allee des trois lavoirs, pessac"),
    ("7 All. Des Trois Lavoirs, Pessac", "7 allee des trois lavoirs, pessac"),
    ("1 bis Allee de la Fontaine, Lege-Cap-Ferret, Nouvelle-Aquitaine", "1 bis allee de la fontaine, lege cap ferret, naq"),
    ("12 Chem. du Moulin, Pornic", "12 chemin du moulin, pornic"),
    ("3 Qu. de la Fosse, Nantes", "3 quai de la fosse, nantes"),
    ("113 R DE L'HOMMELET, Roubaix", "113 rue de l hommelet, roubaix"),
    ("11 R. DE LA GUYAEN, CALAIS, Pas-de-Calais", "11 rue de la guyaen, calais, hdf"),
])
def test_france_addresses(raw, expected):
    out = n.normalize_address(t.transliterate(raw), "France")
    assert out == expected
    assert n.normalize_address(out, "France") == out


@pytest.mark.parametrize("variants", [
    ["13 R. DU PLESSIS, ST-NAZAIRE, Loire-Atlantique", "13 Rue du Plessis, Saint-Nazaire, Pays de la Loire"],
    ["58 R. MARCEAU, TOURCOING, Nord", "58 Rue Marceau, Tourcoing, Hauts-de-France"],
    ["22 Place Dauphine, Merignac, Gironde", "22 PLACE DAUPHINE, MERIGNAC, Nouvelle-Aquitaine"],
])
def test_france_departement_and_region_agree(variants):
    assert len({n.normalize_address(v, "France") for v in variants}) == 1


@pytest.mark.parametrize("region, code", list(FRANCE_REGIONS.items()))
def test_every_france_region_and_departement(region, code):
    assert n.normalize_address(f"1 Rue X, Ville, {region}", "France") == f"1 rue x, ville, {code}"


@pytest.mark.parametrize("country", ["US", "India", "", "Germany"])
def test_france_street_words_only_in_france(country):
    assert n.normalize_address("R K Puram, 5 Imp Road", country) == "r k puram, 5 imp road"


@pytest.mark.parametrize("abbr, full", [
    ("rd", "road"), ("ave", "avenue"), ("dr", "drive"), ("ln", "lane"),
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


@pytest.mark.parametrize("code", sorted(STATE_CODES))
def test_state_codes_kept_as_is(code):
    assert n.normalize_address(f"1 Main St, Town, {code.upper()}") == f"1 main street, town, {code}"


@pytest.mark.parametrize("raw, expected", [
    ("12 Asylum St, Hartford, CT", "12 asylum street, hartford, ct"),        # not "court"
    ("1 Ocean Dr, Miami, FL", "1 ocean drive, miami, fl"),                  # not "floor"
    ("4911 Kessler Ave, Fl 0, Wichita, KS", "4911 kessler avenue, floor 0, wichita, ks"),
    ("51 Main Street, Florida, NY", "51 main street, fl, ny"),
])
def test_state_code_vs_street_word(raw, expected):
    assert n.normalize_address(raw) == expected
    assert n.normalize_address(expected) == expected


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


@pytest.mark.parametrize("raw, expected", [
    ("105 ELM ST, MORGANTON, NC", "105 elm street, morganton, nc"),
    ("105 Elm St NW", "105 elm street nw"),
    ("105 Elm St Apt 5", "105 elm street apartment 5"),
    ("105 Elm St 5", "105 elm street 5"),
    ("105 Elm St E", "105 elm street e"),
    ("St Louis, MO", "saint louis, mo"),
    ("Saint Louis, MO", "saint louis, mo"),
    ("12 St Marks Pl", "12 saint marks place"),
    ("13 Rue St Pierre", "13 rue saint pierre"),
    ("ST-NAZAIRE", "saint nazaire"),
    ("Ste-Foy", "sainte foy"),
    ("Ste Anne Street", "sainte anne street"),
    ("1 Main St, Ste 200", "1 main street, 200"),
    ("1 Main St, Ste B", "1 main street, b"),
    ("1 Main St Ste 200", "1 main street 200"),
])
def test_st_and_ste_by_context(raw, expected):
    assert n.normalize_address(raw) == expected
    assert n.normalize_address(expected) == expected


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


@pytest.mark.parametrize("raw, expected", [
    ("Mumbai, Bombay, MH", "mumbai, mh"),
    ("Office 1, Mumbai City, Bombay, Maharashtra", "office 1, mumbai, mh"),
    ("Kolkata, Calcutta, WB", "kolkata, wb"),
    ("Chennai, Madras, TN", "chennai, tn"),
    ("Bangalore North, Karnataka", "bengaluru north, ka"),
    ("Bengaluru North, Karnataka", "bengaluru north, ka"),
    ("Gurgaon, Haryana", "gurugram, hr"),
    ("Poona, MH", "pune, mh"),
    ("Cochin, Kerala", "kochi, kl"),
    ("Thrissur, Keralam", "thrissur, kl"),
    ("Ahmednagar, Ahilyanagar, MH", "ahmednagar, mh"),
    ("NEWDELHI, Delhi", "new delhi, dl"),
    ("Flat 2D, Kolkata, Kolkata, Howrah, WB", "flat 2d, kolkata, howrah, wb"),
])
def test_india_city_aliases_and_duplicates(raw, expected):
    assert n.normalize_address(raw, "India") == expected
    assert n.normalize_address(expected, "India") == expected


@pytest.mark.parametrize("country", ["US", "France", ""])
def test_india_aliases_only_in_india(country):
    assert n.normalize_address("Bombay Road, Madras Lane", country) == "bombay road, madras lane"


@pytest.mark.parametrize("raw, expected", [
    ("City of El Paso, TX", "el paso, tx"),
    ("CITY OF GREEN BAY, WI", "green bay, wi"),
    ("Town of Islip, NY", "islip, ny"),
    ("Village of Oak Park, IL", "oak park, il"),
    ("Salt Lake City, UT", "salt lake, ut"),
])
def test_city_of(raw, expected):
    assert n.normalize_address(raw) == expected


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


def test_transform_uses_country_column():
    df = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "business_name": ["Martin SA", "Martin SA"],
                       "business_address": ["5 R Hugo", "5 R Hugo"], "country": ["France", "US"]})
    out = n.transform(df)
    assert out.name_norm.tolist() == ["martin", "martin sa"]
    assert out.legal_form.tolist() == ["sa", ""]
    assert out.address_norm.tolist() == ["5 rue hugo", "5 r hugo"]


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
