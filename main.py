"""Entry point: wires the pipeline stages together. Configured by .env; swap an implementation by name there."""

import csv
import os
from pathlib import Path

import pandas as pd

from er.normalize import RuleNormalizer
from er.split import HashSplitter
from er.transliterate import AnyAsciiTransliterator

SPLITS_DIR = Path("data/splits")

TRANSLITERATORS = {"anyascii": AnyAsciiTransliterator}
NORMALIZERS = {"rules": RuleNormalizer}


def load_config(path: Path = Path(".env")) -> dict[str, str]:
    """KEY=VALUE lines from `path` (blank lines and # comments skipped); environment variables override."""
    config = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return {key: os.environ.get(key, value) for key, value in config.items()}


def pick(registry: dict, config: dict[str, str], key: str):
    name = config[key]
    if name not in registry:
        raise SystemExit(f"{key}={name!r} in .env is not one of {sorted(registry)}")
    return registry[name]()


def read_tsv(path: Path, nrows: int | None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, nrows=nrows)


def main():
    config = load_config()
    data_dir, split = Path(config["DATA_DIR"]), config["SPLIT"]
    sets = [int(k) for k in config["SETS"].split(",") if k.strip()]
    nrows = int(config["NROWS"]) if config["NROWS"] else None
    transliterator = pick(TRANSLITERATORS, config, "TRANSLITERATOR")
    normalizer = pick(NORMALIZERS, config, "NORMALIZER")

    if sets and not all((SPLITS_DIR / f"set_{k}").exists() for k in sets):
        print("writing the 10 training sets to", SPLITS_DIR)
        HashSplitter().write(data_dir / "train", SPLITS_DIR)

    def load_source(source: int) -> pd.DataFrame:
        if sets:  # training sets from data/splits/, concatenated
            return pd.concat([read_tsv(SPLITS_DIR / f"set_{k}" / f"source{source}.tsv", nrows) for k in sets],
                             ignore_index=True)
        return read_tsv(data_dir / split / f"{split}_source{source}.tsv", nrows)

    sources = {s: normalizer.transform(transliterator.transform(load_source(s))) for s in (1, 2, 3)}

    for s, df in sources.items():
        print(f"source{s}: {len(df):,} rows")
        print(df[["entity_id", "name_norm", "legal_form", "address_norm"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
