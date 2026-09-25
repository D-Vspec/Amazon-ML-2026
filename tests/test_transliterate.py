from er.transliterate import transliterate


def test_ascii_unchanged():
    assert transliterate("Ram Marketing Pvt Ltd") == "Ram Marketing Pvt Ltd"


def test_accents_stripped():
    assert transliterate("SCI Ptit Àmicale") == "SCI Ptit Amicale"


def test_indic_scripts():
    assert transliterate("ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್") == "guru estet praivet limited"  # Kannada
    assert transliterate("कृष्णा") == "krsna"  # Devanagari


def test_mixed_script():
    assert transliterate("గుజరాత్ Logistics లిమిటెడ్") == "gujrat Logistics limited"  # inherent vowel dropped
