import pandas as pd
import pytest

from er.transliterate import AnyAsciiTransliterator

t = AnyAsciiTransliterator()


# Real "private limited" spellings from train_source2, one per script.
@pytest.mark.parametrize("text, expected", [
    ("प्राइवेट लिमिटेड", "praivet limited"),     # Devanagari
    ("ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "praivet limited"),     # Kannada
    ("ప్రైవేట్ లిమిటెడ్", "praivet limited"),     # Telugu
    ("பிரைவேட் லிமிடெட்", "piraivet limitet"),    # Tamil
    ("প্রাইভেট লিমিটেড", "praibhet limited"),     # Bengali
    ("પ્રાઇવેટ લિમિટેડ", "praivet limited"),      # Gujarati
    ("പ്രൈവറ്റ് ലിമിറ്റഡ്", "praivrr limirrd"),   # Malayalam
    ("ପ୍ରାଇଭେଟ ଲିମିଟେଡ୍", "praibhet limited"),    # Odia
    ("ਪ੍ਰਾਈਵੇਟ ਲਿਮਿਟੇਡ", "praivet limited"),      # Gurmukhi
    ("प्रा. लि.", "pra. li."),                   # Devanagari abbreviation
    ("एलएलपी", "elelpi"),                        # LLP spelled out
])
def test_legal_form_per_script(text, expected):
    assert t.transliterate(text) == expected


# Real Indian state names as they appear in native script in addresses.
@pytest.mark.parametrize("text, expected", [
    ("महाराष्ट्र", "mharastr"),
    ("दिल्ली", "dilli"),
    ("उत्तर प्रदेश", "uttr prdes"),
    ("ಕರ್ನಾಟಕ", "krnatk"),
    ("தமிழ்நாடு", "tmilnatu"),
    ("ગુજરાત", "gujrat"),
    ("পশ্চিমবঙ্গ", "pscimbng"),
    ("తెలంగాణ", "telmgan"),
    ("हरियाणा", "hriyana"),
    ("കേരളം", "kerlm"),
    ("ਪੰਜਾਬ", "pmjab"),
    ("ଓଡ଼ିଶା", "od'isa"),
])
def test_state_names(text, expected):
    assert t.transliterate(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketimg praivet limited"),
    ("ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "guru estet praivet limited"),
    ("కృష్ణా ఇంపెక్స్ లిమిటెడ్", "krsna impeks limited"),
    ("சாவுத் ஐடி", "cavut aiti"),
    ("सॉल्यूशंस", "solyusms"),
    ("मॉडर्न फाइनेंस", "modrn phainems"),
])
def test_full_indic_names(text, expected):
    assert t.transliterate(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("SCI Ptit Àmicale", "SCI Ptit Amicale"),
    ("Engages Àrt Pharmacie SCI", "Engages Art Pharmacie SCI"),
    ("Crêperie Façade", "Creperie Facade"),
    ("Ça va", "Ca va"),
    ("œuvre", "oeuvre"),
    ("Müller", "Muller"),
    ("Ñandú", "Nandu"),
    ("Straße", "Strasse"),
    ("Łódź", "Lodz"),
    ("LLC Moncada Léarning Center", "LLC Moncada Learning Center"),
    ("J 6 PRECISION ÁXIOM", "J 6 PRECISION AXIOM"),
    ("Ínc", "Inc"),
    ("VYAPAR SADAN SEC Â 14", "VYAPAR SADAN SEC A 14"),
    ("PIÑON HILLS", "PINON HILLS"),
])
def test_accented_latin(text, expected):
    assert t.transliterate(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("गुजरात् Logistics लिमिटेड", "gujrat Logistics limited"),
    ("Red मीडिया प्राइवेट लिमिटेड", "Red midiya praivet limited"),
    ("అరిహంత్ Om ఎక్స్‌పోర్ట్స్", "arihmt Om eksports"),
    ("Tech Food પ્રાઇવેટ લિમિટેડ", "Tech Food praivet limited"),
    ("# #163-F , HSIIDC, SECTOR-3, KARNAL, हरियाणा", "# #163-F , HSIIDC, SECTOR-3, KARNAL, hriyana"),
])
def test_mixed_script(text, expected):
    assert t.transliterate(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("—", "-"),
    ("“quoted”", '"quoted"'),
    ("１２３", "123"),       # full-width digits
    ("ｆｕｌｌ", "full"),     # full-width letters
    ("a‍b", "ab"),      # zero-width joiner removed
])
def test_punctuation_and_width(text, expected):
    assert t.transliterate(text) == expected


@pytest.mark.parametrize("text", [
    "",
    " ",
    "Ram Marketing Pvt Ltd",
    "1795 Westchester Drive, High Point, NC",
    "KH NO. -570/13, NEW DELHI",
    "<NULL>",
    "  double  spaces  ",
    "Betsy's Way #3343 (P.C.) [Council] *** & + -- <<",
])
def test_ascii_returned_unchanged(text):
    assert t.transliterate(text) == text


@pytest.mark.parametrize("text", [
    "राम मार्केटिंग", "Àmicale", "ಗುರು", "mixed मीडिया", "１２３",
])
def test_output_is_ascii_and_idempotent(text):
    once = t.transliterate(text)
    assert once.isascii()
    assert t.transliterate(once) == once


def test_case_preserved():
    assert t.transliterate("Àmicale ÀMICALE àmicale") == "Amicale AMICALE amicale"


def _records(**cols):
    base = {"entity_id": ["S2-1"], "business_name": ["x"], "business_address": ["y"], "country": ["India"]}
    base.update(cols)
    return pd.DataFrame(base)


def test_transform_only_touches_text_columns():
    df = _records(business_name=["मॉडर्न"], business_address=["Àrt St"], country=["Índe"])
    out = t.transform(df)
    assert out.iloc[0].tolist() == ["S2-1", "modrn", "Art St", "Índe"]


def test_transform_does_not_mutate_input():
    df = _records(business_name=["मॉडर्न"])
    t.transform(df)
    assert df.business_name[0] == "मॉडर्न"


def test_transform_keeps_rows_index_and_columns():
    df = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S3-3"],
        "business_name": ["राम", "Ram", ""],
        "business_address": ["", "Àrt St", "दिल्ली"],
        "country": ["India", "US", "France"],
    }, index=[10, 20, 30])
    out = t.transform(df)
    assert list(out.columns) == list(df.columns)
    assert list(out.index) == [10, 20, 30]
    assert out.business_name.tolist() == ["ram", "Ram", ""]
    assert out.business_address.tolist() == ["", "Art St", "dilli"]


def test_transform_empty_frame():
    df = _records().iloc[:0]
    assert t.transform(df).empty


def test_transform_keeps_extra_columns():
    df = _records(business_name=["राम"]).assign(extra=[1])
    assert t.transform(df).extra.tolist() == [1]
