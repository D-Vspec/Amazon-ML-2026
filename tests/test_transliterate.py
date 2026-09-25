import pandas as pd

from er.transliterate import AnyAsciiTransliterator

t = AnyAsciiTransliterator()


def test_ascii_unchanged():
    assert t.transliterate("Ram Marketing Pvt Ltd") == "Ram Marketing Pvt Ltd"


def test_accents_stripped():
    assert t.transliterate("SCI Ptit Àmicale") == "SCI Ptit Amicale"


def test_indic_scripts():
    assert t.transliterate("ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್") == "guru estet praivet limited"  # Kannada
    assert t.transliterate("कृष्णा") == "krsna"  # Devanagari


def test_mixed_script():
    assert t.transliterate("గుజరాత్ Logistics లిమిటెడ్") == "gujrat Logistics limited"  # inherent vowel dropped


def test_transform_only_touches_text_columns():
    df = pd.DataFrame({"entity_id": ["S2-1"], "business_name": ["मॉडर्न"],
                       "business_address": ["Àrt St"], "country": ["India"]})
    out = t.transform(df)
    assert out.iloc[0].tolist() == ["S2-1", "modrn", "Art St", "India"]
    assert df.business_name[0] == "मॉडर्न"  # input not mutated
