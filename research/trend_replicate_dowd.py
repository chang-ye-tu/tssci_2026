#!/usr/bin/env python3
"""External check: reproduce published life-expectancy deficits with this code.

Dowd et al. (2026, medRxiv) fit a Poisson Lee-Carter model to both sexes
combined, 2000-2019, forecast 2020-2024 from the fitted level and report the
deficit in life expectancy at birth.  Their Taiwan series is
+0.39, -0.25, -1.36, -1.27, -1.04 (2020-2024); Japan -0.80, -0.91, -1.07
(2022-2024).

This script applies the same specification with the functions of trendlib.py
to the HMD data used in this study (their inputs were STMF deaths and WPP
populations, so agreement need not be exact).

Run from the repository root:  python3 research/trend_replicate_dowd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trendlib as tl  # noqa: E402

OUT = Path("results_trend")
PUBLISHED = {"TWN": {2020: 0.39, 2021: -0.25, 2022: -1.36, 2023: -1.27, 2024: -1.04}, "JPN": {2022: -0.80, 2023: -0.91, 2024: -1.07}}


def main() -> None:
    rows = []
    for pop in ("TWN", "JPN"):
        deaths = tl.read_hmd_table(pop, "Deaths")
        expos = tl.read_hmd_table(pop, "Exposures")
        D = tl.age_year_matrix(deaths, "Total", range(2000, 2025), 0, 100)
        E = tl.age_year_matrix(expos, "Total", range(2000, 2025), 0, 100)
        fc = tl.forecast(D[:, :20], E[:, :20], np.arange(1, 6), jump="fit")
        for h, year in enumerate(range(2020, 2025)):
            observed = float(tl.life_expectancy(D[:, 20 + h] / E[:, 20 + h]))
            expected = float(tl.life_expectancy(fc["central"][:, h]))
            rows.append((pop, year, observed, expected, observed - expected, PUBLISHED[pop].get(year, np.nan)))
    out = pd.DataFrame(rows, columns=["population", "year", "e0_observed", "e0_expected", "deficit_this_study", "deficit_dowd_et_al"])
    out["difference"] = out["deficit_this_study"] - out["deficit_dowd_et_al"]
    out.to_csv(OUT / "replication_dowd.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
