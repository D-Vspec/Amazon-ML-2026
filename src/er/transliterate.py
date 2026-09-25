"""Convert any script (Devanagari, Tamil, accented Latin, ...) to plain ASCII."""

from anyascii import anyascii


def transliterate(text: str) -> str:
    # Most records are already ASCII; skip the per-character lookup for them.
    if text.isascii():
        return text
    return anyascii(text)
