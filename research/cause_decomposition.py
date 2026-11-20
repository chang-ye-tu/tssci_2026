#!/usr/bin/env python3
"""Age-by-cause decomposition of changes in Taiwan's adult life table.

All-cause death rates by single year of age come from the HMD (2008-2024) and
from the rebuilt 2025 rates (research/taiwan_2025.py).  Within each five-year
age group they are split by cause in proportion to the Ministry of Health and
Welfare's deaths by cause, so the all-cause life table is unchanged.

Because the cause-selection rules changed in 2019, changes are decomposed
within 2008-2018 or within 2019-2025 only:

    2012 -> 2018   six pre-pandemic years (earlier rules)
    2019 -> 2025   six years from the last pre-pandemic year (2016 rules)
    2019 -> 2022 and 2022 -> 2025   shock and recovery

Method: Horiuchi, Wilmoth & Pletcher (2008), straight path between the two
rate schedules.  The contribution of cause c at age x equals the path-average
derivative of the measure with respect to the death rate at age x times the
change in the cause-specific rate.

Run from the repository root:  python3 research/cause_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trendlib as tl  # noqa: E402

OUT = Path("results_cause")
AGE0, OPEN_AGE = 30, 100
AGES = np.arange(AGE0, OPEN_AGE + 1)
GROUP_OF_AGE = np.minimum((AGES - AGE0) // 5, (OPEN_AGE - AGE0) // 5)          # 0..14
GROUP_LOWER = [30 + 5 * g for g in range(15)]
GROUP_LABELS = [f"{a}-{a + 4}" for a in range(30, 100, 5)] + ["100+"]
BANDS = {"30-49": (30, 49), "50-64": (50, 64), "65-79": (65, 79), "80+": (80, 200)}
PERIODS = [(2012, 2018), (2019, 2025), (2019, 2022), (2022, 2025), (2019, 2024)]
MEASURES = ("e", "q25", "q50", "q75")
YEARS = range(2008, 2026)
STEPS = 60


def all_cause_rates(sex: str) -> dict[int, np.ndarray]:
    deaths = tl.read_hmd_table("TWN", "Deaths")
    expos = tl.read_hmd_table("TWN", "Exposures")
    D = tl.age_year_matrix(deaths, sex, range(2008, 2025), AGE0, OPEN_AGE)
    E = tl.age_year_matrix(expos, sex, range(2008, 2025), AGE0, OPEN_AGE)
    rates = {year: D[:, i] / E[:, i] for i, year in enumerate(range(2008, 2025))}
    exposures = {year: E[:, i] for i, year in enumerate(range(2008, 2025))}
    r25 = pd.read_csv("results_trend/taiwan_2025_rates.csv")
    r25 = r25[r25.sex == sex].sort_values("age")
    rates[2025] = r25["rate"].to_numpy()
    exposures[2025] = r25["exposure"].to_numpy()
    return rates, exposures


def main() -> None:
    OUT.mkdir(exist_ok=True)
    by_cause = pd.read_csv(OUT / "deaths_by_cause.csv")
    by_cause = by_cause[by_cause.age_lower >= AGE0]
    causes = sorted(by_cause.cause_group.unique())
    contrib_rows, rate_rows, check_rows = [], [], []

    for sex in tl.SEXES:
        m_all, expo = all_cause_rates(sex)
        shares = {}
        for year in YEARS:
            d = (by_cause[(by_cause.sex == sex) & (by_cause.year == year)]
                 .pivot(index="age_lower", columns="cause_group", values="deaths").reindex(index=GROUP_LOWER, columns=causes).fillna(0.0))
            tot = d.sum(axis=1).to_numpy()
            if (tot <= 0).any():
                raise ValueError(f"no deaths in an age group: {sex} {year}")
            shares[year] = d.to_numpy() / tot[:, None]                              # (15 groups, C)
        cause_rates = {year: m_all[year][:, None] * shares[year][GROUP_OF_AGE, :] for year in YEARS}   # (A, C)

        # descriptive: cause-specific death rates by age band, standardised to the 2019 exposure of this sex
        w19 = expo[2019]
        for band, (lo, hi) in BANDS.items():
            idx = (AGES >= lo) & (AGES <= hi)
            w = w19[idx] / w19[idx].sum()
            for year in YEARS:
                std = (cause_rates[year][idx, :] * w[:, None]).sum(axis=0) * 1e5
                for c, name in enumerate(causes):
                    rate_rows.append((sex, band, year, name, std[c]))

        for y0, y1 in PERIODS:
            m0, m1 = m_all[y0], m_all[y1]
            dR = cause_rates[y1] - cause_rates[y0]                                  # (A, C)
            for name in MEASURES:
                func = lambda x, n=name: tl.measure(x, n, AGE0)
                S = tl.sensitivity_integral(func, m0, m1, STEPS)                    # (A,)
                contrib = S[:, None] * dR                                           # (A, C)
                total = float(func(m1[None, :])[0] - func(m0[None, :])[0])
                check_rows.append((sex, y0, y1, name, total, float(contrib.sum()), float(contrib.sum() - total)))
                grouped = np.zeros((15, len(causes)))
                np.add.at(grouped, GROUP_OF_AGE, contrib)
                for g, label in enumerate(GROUP_LABELS):
                    for c, cname in enumerate(causes):
                        contrib_rows.append((sex, y0, y1, name, label, GROUP_LOWER[g], cname, grouped[g, c]))

    contrib = pd.DataFrame(contrib_rows, columns=["sex", "from_year", "to_year", "measure", "age_group", "age_lower", "cause_group", "contribution"])
    contrib.to_csv(OUT / "cause_contributions.csv", index=False)
    pd.DataFrame(rate_rows, columns=["sex", "age_band", "year", "cause_group", "rate_per_100k"]).to_csv(OUT / "cause_rates_by_band.csv", index=False)
    checks = pd.DataFrame(check_rows, columns=["sex", "from_year", "to_year", "measure", "total_change", "sum_of_contributions", "residual"])
    checks.to_csv(OUT / "decomposition_checks.csv", index=False)

    def band_of(lower: int) -> str:
        return next(b for b, (lo, hi) in BANDS.items() if lo <= lower <= hi)

    contrib["age_band"] = contrib["age_lower"].map(band_of)
    summary = contrib.groupby(["sex", "from_year", "to_year", "measure", "age_band", "cause_group"], as_index=False)["contribution"].sum()
    summary.to_csv(OUT / "cause_contributions_by_band.csv", index=False)

    meta = {"script": "research/cause_decomposition.py", "periods": PERIODS, "measures": list(MEASURES), "steps": STEPS,
            "max_abs_residual_years": float(checks["residual"].abs().max()),
            "note": "Compare within 2008-2018 or within 2019-2025 only (cause-selection rules changed in 2019)."}
    (OUT / "decomposition_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("max |sum of contributions - total change| =", round(meta["max_abs_residual_years"], 5), "years")
    print(checks[checks.measure == "e"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
