"""Invariants that must hold for *any* input string, checked on generated text."""

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from er.normalize import COUNTRY_LEGAL_FORMS, LEGAL_FORMS, STATES, RuleNormalizer
from er.transliterate import AnyAsciiTransliterator

n = RuleNormalizer()
t = AnyAsciiTransliterator()

CANONICAL_LEGAL = set(LEGAL_FORMS.values()) | {v for forms in COUNTRY_LEGAL_FORMS.values() for v in forms.values()}
NAME_TOKEN = re.compile(r"^[a-z0-9]+$")
ADDRESS_TOKEN = re.compile(r"^[a-z0-9]+(?:[/-][a-z0-9]+)*$")

# Text biased toward the characters and words that trigger rules.
VOCAB = ["pvt", "ltd", "private", "limited", "llc", "l.l.c.", "inc", "co", "company", "dba", "trading as",
         "t/a", "d/b/a", "&", "+", "and", "pra.", "li.", "elelpi", ".com", "www.", ".co.in", "0", "1", "00",
         "st", "rd", "city", "unit", "#", "null", "<NULL>", "n/a", "-", "/", ",", ".", " ", "  ", "(", ")",
         "[", "]", "***", "kansas", "new york", "delhi", "dilli", "mharastr", "od'isa", "a", "b", "z",
         "Ram", "Kansas City", "Àmicale", "राम", "ಗುರು", "\t", "‍", "İ", "ß",
         "r", "imp", "ndeg", "N°", "ste", "st", "saint", "nord", "gironde", "ct", "fl", "florida", "d.b.a.",
         "l.l.c.", "pvt.ltd", "sa", "ei", "et", "cie", "(india)", "(france)", "bombay", "keralam"]
countries = st.sampled_from(["US", "India", "France", "", "Germany"])
tricky_text = st.lists(st.sampled_from(VOCAB), max_size=12).map("".join)
any_text = st.one_of(st.text(max_size=60), tricky_text)
settings.register_profile("thorough", max_examples=2000, deadline=None)
settings.load_profile("thorough")


def assert_clean_spacing(s):
    assert s == s.strip()
    assert "  " not in s


@given(any_text, countries)
def test_name_output_shape(raw, country):
    core, legal = n.normalize_name(raw, country)
    assert_clean_spacing(core)
    assert all(NAME_TOKEN.match(tok) for tok in core.split())
    legal_tokens = legal.split()
    assert set(legal_tokens) <= CANONICAL_LEGAL
    assert legal_tokens == sorted(set(legal_tokens))


@given(any_text, countries)
def test_name_core_idempotent(raw, country):
    core, _ = n.normalize_name(raw, country)
    assert n.normalize_name(core, country)[0] == core


@given(any_text)
def test_name_empty_only_when_input_has_no_letters_or_digits(raw):
    core, legal = n.normalize_name(t.transliterate(raw))
    if core == "":
        assert legal == ""


@given(any_text, countries)
def test_address_output_shape(raw, country):
    out = n.normalize_address(raw, country)
    if out == "":
        return
    for component in out.split(", "):
        assert component
        assert_clean_spacing(component)
        for tok in component.split():
            assert ADDRESS_TOKEN.match(tok), tok
            if tok.isdigit():
                assert tok == "0" or not tok.startswith("0")


@given(any_text, countries)
def test_address_idempotent(raw, country):
    once = n.normalize_address(raw, country)
    assert n.normalize_address(once, country) == once


@given(any_text)
def test_transliterated_output_is_ascii(raw):
    core, legal = n.normalize_name(t.transliterate(raw))
    assert (core + legal + n.normalize_address(t.transliterate(raw))).isascii()


@given(st.sampled_from(sorted(STATES)), any_text)
def test_state_component_always_becomes_code(state, prefix):
    out = n.normalize_address(f"{prefix}, {state}")
    assert out.endswith(STATES[state])
