"""Before/after samples of the cleaning stage -> output/clean_samples.md

US + India from the train sources, France from the test sources (France is not in train).
Deterministic: per file and country, take the first POOL rows, then every STRIDE-th one.
Also reports component-extraction coverage over each pool and cleaning throughput.

Run from the repo root:  .venv/Scripts/python notebooks/sample_clean.py
"""

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from clean import clean_record  # noqa: E402

DATA = ROOT / "6ab10eb3b23ba_student_resource/student_resource/dataset"
FILES = [("train", n, ["US", "India"]) for n in (1, 2, 3)] + [("test", n, ["France"]) for n in (1, 2, 3)]
POOL, STRIDE = 2000, 100
MAX_SCAN = 3_000_000  # rows scanned per file while filling pools

csv.field_size_limit(10**7)


def pools_for(path: Path, countries: list[str]) -> dict[str, list[dict]]:
    pools = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
            if r["country"] in countries and len(pools[r["country"]]) < POOL:
                pools[r["country"]].append(r)
            if i >= MAX_SCAN or all(len(pools[c]) >= POOL for c in countries):
                break
    return pools


def cell(s: str) -> str:
    return (s or "").replace("|", "\\|") or "·"


def main():
    lines = ["# Cleaning samples", "",
             f"Per file and country: first {POOL} rows, every {STRIDE}th shown. `·` = empty.", ""]
    coverage, n_clean, t_clean = [], 0, 0.0
    for split, n, countries in FILES:
        path = DATA / split / f"{split}_source{n}.tsv"
        pools = pools_for(path, countries)
        for country in countries:
            pool = pools[country]
            t0 = time.perf_counter()
            cleaned = [clean_record(r) for r in pool]
            t_clean += time.perf_counter() - t0
            n_clean += len(pool)
            k = len(pool)
            coverage.append((path.name, country, k,
                             *(sum(bool(c[f]) for c in cleaned) / k if k else 0
                               for f in ("postal_code", "state", "city"))))
            lines += [f"## {path.name} — {country} ({len(cleaned[::STRIDE])} of {k})", "",
                      "| id | raw name | clean name | raw address | clean address | postal | state | city |",
                      "|---|---|---|---|---|---|---|---|"]
            for c in cleaned[::STRIDE]:
                lines.append("| " + " | ".join(cell(c[f]) for f in (
                    "entity_id", "business_name", "name_clean", "business_address", "address_clean",
                    "postal_code", "state", "city")) + " |")
            lines.append("")
    lines += ["## Component coverage (share of pool with a non-empty field)", "",
              "| file | country | rows | postal | state | city |", "|---|---|---|---|---|---|"]
    lines += [f"| {f} | {c} | {k} | {p:.1%} | {s:.1%} | {ci:.1%} |" for f, c, k, p, s, ci in coverage]
    lines += ["", f"Throughput: {n_clean} records in {t_clean:.2f}s = {n_clean / t_clean:,.0f} records/s"]
    out = ROOT / "output" / "clean_samples.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines[-len(coverage) - 4:]))


if __name__ == "__main__":
    main()
