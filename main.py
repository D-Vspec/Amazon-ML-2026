"""Entry point: wires the pipeline stages together. Swap an implementation by name on the CLI."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from er.normalize import RuleNormalizer
from er.split import HashSplitter
from er.transliterate import AnyAsciiTransliterator

DATA_DIR = Path("6ab10eb3b23ba_student_resource/student_resource/dataset")
SPLITS_DIR = Path("data/splits")

TRANSLITERATORS = {"anyascii": AnyAsciiTransliterator}
NORMALIZERS = {"rules": RuleNormalizer}


def read_tsv(path: Path, nrows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=nrows)


def load_source(args, source: int) -> pd.DataFrame:
    if args.sets:  # training sets from data/splits/, concatenated
        return pd.concat([read_tsv(SPLITS_DIR / f"set_{k}" / f"source{source}.tsv", args.nrows) for k in args.sets],
                         ignore_index=True)
    return read_tsv(args.data_dir / args.split / f"{args.split}_source{source}.tsv", args.nrows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--nrows", type=int, help="rows per source file (for quick runs)")
    parser.add_argument("--sets", type=int, nargs="+", choices=range(HashSplitter.n_sets),
                        help="run on these training sets from data/splits/ (see docs/data_splits.md)")
    parser.add_argument("--make-splits", action="store_true", help="write the 10 training sets to data/splits/ and exit")
    parser.add_argument("--transliterator", choices=TRANSLITERATORS, default="anyascii")
    parser.add_argument("--normalizer", choices=NORMALIZERS, default="rules")
    args = parser.parse_args()

    if args.make_splits:
        for k, files in HashSplitter().write(args.data_dir / "train", SPLITS_DIR).items():
            print(f"set_{k}", files)
        return

    transliterator = TRANSLITERATORS[args.transliterator]()
    normalizer = NORMALIZERS[args.normalizer]()

    sources = {s: load_source(args, s) for s in (1, 2, 3)}
    sources = {s: normalizer.transform(transliterator.transform(df)) for s, df in sources.items()}

    for s, df in sources.items():
        print(f"source{s}: {len(df):,} rows")
        print(df[["entity_id", "name_norm", "legal_form", "address_norm"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
