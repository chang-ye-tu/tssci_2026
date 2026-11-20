#!/usr/bin/env python3
"""Compare the results recomputed by reproduce.py with the published results in published/.

For every file in published/ the script looks at its recomputed counterpart:

    published/results_trend/*   <->  results_trend/*
    published/results_cause/*   <->  results_cause/*
    published/generated/*.json  <->  manuscript/generated/*.json
    published/figures/*.png     <->  manuscript/figures/*.png

A file passes if it is identical byte for byte.  Some files can differ in bytes although their content is the
same, and are compared by content instead: metadata (JSON files that record the date and the run time of a
script) and gzip files (the header holds a time stamp).  Figures in PDF hold a creation date and are not
compared; the PNG versions are.  If a table of numbers differs, the script says how many values differ and by
how much, because another version of numpy or another processor can change the last digits.  A figure that
differs is reported as a warning: it depends on the fonts installed and is not a result.

Before comparing, the script checks with CHECKSUMS.sha256 that the data, the programs and the published results
in this folder are the files the package was shipped with.

reproduce.py runs this script as its last step.  To run it alone, from the package root:  python3 verify.py
Exit status 0: every result reproduces.  1: something is missing or differs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

PAIRS = (
    ("published/results_trend", "results_trend"),
    ("published/results_cause", "results_cause"),
    ("published/generated", "manuscript/generated"),
    ("published/figures", "manuscript/figures"),
)
VOLATILE_KEYS = {"created_utc", "seconds"}        # date and run time written into the metadata
NOT_RECOMPUTED = {"numbers_used.csv"}             # written when the article is typeset; kept for reference
SHIPPED = ("data/", "research/", "published/", "online_appendix/")
CLOSE = 1e-8                                      # largest relative difference still counted as rounding


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_package() -> list[str]:
    """Files of the package that are missing or no longer the shipped ones."""
    problems, n = [], 0
    for line in Path("CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if not name.startswith(SHIPPED):
            continue
        n += 1
        path = Path(name)
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256(path) != digest:
            problems.append(f"changed since the package was made: {name}")
    print(f"package: {n} files of data, programs and published results checked, {len(problems)} problems")
    return problems


def without_volatile(obj):
    if isinstance(obj, dict):
        return {k: without_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [without_volatile(v) for v in obj]
    return obj


def leaves(obj, prefix=""):
    """Flatten nested JSON into {path: value}."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}/{i}")
    else:
        yield prefix, obj


def compare_json(published: Path, recomputed: Path) -> tuple[str, str]:
    a = without_volatile(json.loads(published.read_text(encoding="utf-8")))
    b = without_volatile(json.loads(recomputed.read_text(encoding="utf-8")))
    if a == b:
        return "same content", "identical apart from the date and the run time"
    fa, fb = dict(leaves(a)), dict(leaves(b))
    differing = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    examples = "; ".join(f"{k}: {fa.get(k)!r} -> {fb.get(k)!r}" for k in differing[:3])
    return "DIFFERENT", f"{len(differing)} of {len(fa)} values differ, e.g. {examples}"


def compare_csv(published: Path, recomputed: Path) -> tuple[str, str]:
    import numpy as np
    import pandas as pd
    a, b = pd.read_csv(published), pd.read_csv(recomputed)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return "DIFFERENT", f"shape {a.shape} with columns {list(a.columns)[:4]}... against {b.shape}"
    text_cells, worst, n_num = 0, 0.0, 0
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
            both_nan = np.isnan(x) & np.isnan(y)
            differ = ~both_nan & (x != y)
            n_num += int(differ.sum())
            if differ.any():
                scale = np.maximum(np.maximum(np.abs(x[differ]), np.abs(y[differ])), 1e-12)
                rel = np.abs(x[differ] - y[differ]) / scale
                worst = max(worst, float(np.nanmax(rel)) if np.isfinite(rel).any() else float("inf"))
        else:
            text_cells += int((a[col].astype(str) != b[col].astype(str)).sum())
    if text_cells == 0 and n_num == 0:
        return "same content", "same values, written differently"
    detail = f"{n_num} numbers differ (largest relative difference {worst:.2e}), {text_cells} other cells differ"
    return ("numerically close" if text_cells == 0 and worst <= CLOSE else "DIFFERENT"), detail


def compare(published: Path, recomputed: Path) -> tuple[str, str]:
    if published.read_bytes() == recomputed.read_bytes():
        return "identical", ""
    name = published.name
    if name.endswith(".json"):
        return compare_json(published, recomputed)
    if name.endswith(".gz"):
        same = gzip.decompress(published.read_bytes()) == gzip.decompress(recomputed.read_bytes())
        if same:
            return "same content", "identical after decompression"
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:           # compare the tables inside
            pa, pb = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            pa.write_bytes(gzip.decompress(published.read_bytes()))
            pb.write_bytes(gzip.decompress(recomputed.read_bytes()))
            return compare_csv(pa, pb)
    if name.endswith(".csv"):
        return compare_csv(published, recomputed)
    if name.endswith(".png"):
        return "figure differs", "the picture is not identical (fonts or matplotlib version); not a result"
    return "DIFFERENT", "bytes differ"


def main() -> None:
    problems = check_package()
    for p in problems[:20]:
        print("  " + p)
    counts: dict[str, int] = {}
    report: list[str] = []
    for pub_dir, new_dir in PAIRS:
        for published in sorted(Path(pub_dir).iterdir()):
            if not published.is_file() or published.name in NOT_RECOMPUTED or published.suffix == ".pdf":
                continue
            recomputed = Path(new_dir) / published.name
            if not recomputed.is_file():
                status, detail = "MISSING", "not recomputed; run reproduce.py first"
            else:
                status, detail = compare(published, recomputed)
            counts[status] = counts.get(status, 0) + 1
            if status != "identical":
                report.append(f"  {status:17s} {recomputed}: {detail}")
    total = sum(counts.values())
    print(f"results: {total} files compared with published/")
    for status in ("identical", "same content", "numerically close", "figure differs", "DIFFERENT", "MISSING"):
        if counts.get(status):
            print(f"  {counts[status]:3d} {status}")
    if report:
        print("details:")
        print("\n".join(report))
    failed = bool(problems) or any(counts.get(s) for s in ("DIFFERENT", "MISSING"))
    if failed:
        print("RESULT: NOT reproduced; see the lines above")
        sys.exit(1)
    if counts.get("numerically close"):
        print("RESULT: every result reproduces up to rounding in the last digits")
    else:
        print("RESULT: every result reproduces")


if __name__ == "__main__":
    main()
