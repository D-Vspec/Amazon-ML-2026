"""Entry point: wires the pipeline stages together. Swap an implementation by name on the CLI."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from er.transliterate import AnyAsciiTransliterator

DATA_DIR = Path("6ab10eb3b23ba_student_resource/student_resource/dataset")

TRANSLITERATORS = {"anyascii": AnyAsciiTransliterator}


def load_source(data_dir: Path, split: str, source: int, nrows: int | None) -> pd.DataFrame:
    return pd.read_csv(data_dir / split / f"{split}_source{source}.tsv", sep="\t", dtype=str,
                       keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=nrows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--nrows", type=int, help="rows per source file (for quick runs)")
    parser.add_argument("--transliterator", choices=TRANSLITERATORS, default="anyascii")
    args = parser.parse_args()

    transliterator = TRANSLITERATORS[args.transliterator]()

    sources = {s: load_source(args.data_dir, args.split, s, args.nrows) for s in (1, 2, 3)}
    sources = {s: transliterator.transform(df) for s, df in sources.items()}

    for s, df in sources.items():
        print(f"source{s}: {len(df):,} rows")
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
