"""Random spot check: raw -> cleaned for a few rows per country.

Seeded random byte offsets into the source files (a different selection from
sample_clean.py, which takes every 100th row from the head of each file).
US + India from train sources, France from test sources (France is not in train).

Run from the repo root:  .venv/Scripts/python notebooks/spot_check.py [seed]
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from clean import clean_record  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset"
QUOTAS = {"US": ("train", 7), "India": ("train", 7), "France": ("test", 6)}
COLS = ["entity_id", "business_name", "business_address", "country"]


def random_row(rng: random.Random, path: Path) -> dict:
    """Row starting after a random byte offset (skip the partial line we land in)."""
    with open(path, "rb") as f:
        f.seek(rng.randrange(path.stat().st_size))
        f.readline()
        line = f.readline().decode("utf-8").rstrip("\r\n")
    return dict(zip(COLS, line.split("\t")))


def main(seed: int):
    rng = random.Random(seed)
    for country, (split, n) in QUOTAS.items():
        files = [DATA / split / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
        rows, seen = [], set()
        while len(rows) < n:
            r = random_row(rng, rng.choice(files))
            if r.get("country") == country and r["entity_id"] not in seen:
                seen.add(r["entity_id"])
                rows.append(clean_record(r))
        print(f"\n### {country}")
        for r in rows:
            print(f"- {r['entity_id']}\n"
                  f"    name: {r['business_name']!r} -> {r['name_clean']!r}\n"
                  f"    addr: {r['business_address']!r} -> {r['address_clean']!r}\n"
                  f"    postal={r['postal_code']!r} state={r['state']!r} city={r['city']!r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20260925)
