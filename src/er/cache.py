"""On-disk cache of blocking + featurizing output per set, so experiments don't re-block (~10 min per set).

Layout: <root>/<key>/sets_<k1_k2..>/{pairs,s1,targets}.parquet plus a `done` marker written last, so an
interrupted write is never read back. <key> is a hash of every setting that changes the pairs or features.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

# Bump when blocker or featurizer code changes in a way the settings in the key don't capture.
CACHE_VERSION = 1
RECORD_COLUMNS = ["entity_id", "raw_name", "raw_address"]


class PairCache:
    def __init__(self, root: Path, settings: dict):
        key = hashlib.sha1(json.dumps({"version": CACHE_VERSION, **settings}, sort_keys=True).encode()).hexdigest()
        self.dir = Path(root) / key[:12]

    def _path(self, set_list: list[int]) -> Path:
        return self.dir / ("sets_" + "_".join(map(str, set_list)))

    def load(self, set_list: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        """(pairs, s1 records, target records) if cached, else None."""
        path = self._path(set_list)
        if not (path / "done").exists():
            return None
        return tuple(pd.read_parquet(path / f"{name}.parquet") for name in ("pairs", "s1", "targets"))

    def load_column(self, set_list: list[int], name: str) -> pd.Series | None:
        """An extra per-pair column (in pairs.parquet's row order) saved with save_column, else None."""
        path = self._path(set_list) / f"column_{name}.parquet"
        return pd.read_parquet(path)[name] if path.exists() else None

    def save_column(self, set_list: list[int], name: str, values) -> None:
        """Cache an expensive per-pair column (e.g. cross-encoder scores) alongside the set's pairs."""
        path = self._path(set_list) / f"column_{name}.parquet"
        pd.DataFrame({name: values}).to_parquet(path.with_suffix(".tmp"), index=False)
        path.with_suffix(".tmp").rename(path)  # atomic: a half-written file is never read

    def save(self, set_list: list[int], pairs: pd.DataFrame, s1: pd.DataFrame, targets: pd.DataFrame) -> None:
        path = self._path(set_list)
        path.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(path / "pairs.parquet", index=False)
        s1[RECORD_COLUMNS].to_parquet(path / "s1.parquet", index=False)
        targets[RECORD_COLUMNS].to_parquet(path / "targets.parquet", index=False)
        (path / "done").touch()
