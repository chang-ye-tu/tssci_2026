#!/usr/bin/env python3
"""Did changes in household registration distort Taiwan's death rates?

Taiwan's population figures are registered populations.  A person who stays
abroad for two years is removed from the register and is entered again on
return.  During the pandemic many people abroad were removed (2021-2022) and
many came back on the register in 2023.  If the people who move on and off the
register rarely die in Taiwan, their removal lowers the denominator of the
death rate without changing the numerator.

This script measures how large that movement is.  For each sex and cohort,

    net registration change = P(x+1, end of year t) - P(x, end of year t-1) + deaths of the cohort in year t

where the cohort's deaths are approximated by half of the deaths at age x and
half of those at age x+1 in year t.  The result is summed over age bands and
expressed per 100 of the population at the start of the year, and accumulated
from the end of 2019.

Run from the repository root (reads downloaded workbooks, hence -I):
    python3 -I research/taiwan_registration_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taiwan_2025 as t25  # noqa: E402
import trendlib as tl  # noqa: E402

OUT = Path("results_trend")
YEARS = range(2012, 2026)
BANDS = {"30-49": (30, 49), "50-64": (50, 64), "65-79": (65, 79), "80-94": (80, 94)}


def main() -> None:
    deaths = t25.read_ris(t25.DEATHS_XLS, "死亡單齡")
    pop = t25.read_ris(t25.POP_XLS, "單齡")
    rows = []
    for sex in tl.SEXES:
        P = pop[pop.sex == sex].pivot(index="age", columns="year", values="value")
        D = deaths[deaths.sex == sex].pivot(index="age", columns="year", values="value")
        for band, (lo, hi) in BANDS.items():
            cumulative = 0.0
            for year in YEARS:
                ages = np.arange(lo, hi + 1)
                start = P.loc[ages, year - 1].to_numpy()
                end = P.loc[ages + 1, year].to_numpy()
                cohort_deaths = 0.5 * D.loc[ages, year].to_numpy() + 0.5 * D.loc[ages + 1, year].to_numpy()
                net = float((end - start + cohort_deaths).sum())
                rate = 100.0 * net / float(start.sum())
                if year >= 2020:
                    cumulative += rate
                rows.append((sex, band, year, float(start.sum()), net, rate, cumulative if year >= 2020 else np.nan))
    out = pd.DataFrame(rows, columns=["sex", "age_band_at_start", "year", "population_at_start", "net_registration_change",
                                      "per_100_population", "cumulative_since_end_2019_per_100"])
    out.to_csv(OUT / "taiwan_registration_check.csv", index=False)
    pd.set_option("display.width", 220)
    print("net registration change per 100 population at the start of the year")
    print(out.pivot_table(index=["sex", "age_band_at_start"], columns="year", values="per_100_population").round(2).to_string())
    print("\ncumulative since the end of 2019, per 100")
    print(out[out.year >= 2020].pivot_table(index=["sex", "age_band_at_start"], columns="year", values="cumulative_since_end_2019_per_100").round(2).to_string())


if __name__ == "__main__":
    main()
