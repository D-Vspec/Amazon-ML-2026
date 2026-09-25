"""Convert any script (Devanagari, Tamil, accented Latin, ...) to plain ASCII."""

import pandas as pd
from anyascii import anyascii

TEXT_COLUMNS = ["business_name", "business_address"]


class AnyAsciiTransliterator:
    def transliterate(self, text: str) -> str:
        # Most records are already ASCII; skip the per-character lookup for them.
        if text.isascii():
            return text
        return anyascii(text)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        records = records.copy()
        for col in TEXT_COLUMNS:
            records[col] = records[col].map(self.transliterate)
        return records
