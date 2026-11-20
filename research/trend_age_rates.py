#!/usr/bin/env python3
"""Death rates by age group against the pre-pandemic trend, East Asia.

The age decomposition (research/trend_baseline.py) expresses each age group's
part of the gap in years of life expectancy.  This script states the same
comparison as death rates, which are easier to read:

    ratio = observed death rate / death rate expected from the trend

for the four age bands used in the paper and for five-year age groups, in
Taiwan, Japan, South Korea and Hong Kong, 2020-2024 (Taiwan also 2025, from
the rebuilt rates of research/taiwan_2025.py).  Rates are standardised within
each age group to the 2019 exposure of the same population and sex.

The expected rates come from the main specification (Poisson Lee-Carter,
2003-2019, forecast anchored to the 2017-2019 level).  The 95% interval of
the ratio combines refitting on resampled data, drift uncertainty, random-walk
innovations and Poisson variation of the target-year deaths, exactly as for
the life-table measures.

The script also records the model-free comparison with 2019 (rate in year t
divided by the rate in 2019).

Run from the repository root:  python3 research/trend_age_rates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trendlib as tl  # noqa: E402

OUT = Path("results_trend")
AGE0, OPEN_AGE = 30, 100
AGES = np.arange(AGE0, OPEN_AGE + 1)
FOCUS = ("TWN", "JPN", "KOR", "HKG")
FIT = (2003, 2019)
N_BOOT, N_PATHS, SEED = 2000, 4, 20261120
MCSE_SEED = 20261198            # separate stream for the Monte Carlo error of interval end points
BANDS = {"30-49": (30, 49), "50-64": (50, 64), "65-79": (65, 79), "80+": (80, 200)}
FIVE = {f"{a}-{a + 4}": (a, a + 4) for a in range(30, 95, 5)}
FIVE["95+"] = (95, 200)


def main() -> None:
    rng = np.random.default_rng(SEED)
    r25 = pd.read_csv(OUT / "taiwan_2025_rates.csv")
    rows = []
    for pop in FOCUS:
        deaths = tl.read_hmd_table(pop, "Deaths")
        expos = tl.read_hmd_table(pop, "Exposures")
        for sex in tl.SEXES:
            D = tl.age_year_matrix(deaths, sex, range(2003, 2025), AGE0, OPEN_AGE)
            E = tl.age_year_matrix(expos, sex, range(2003, 2025), AGE0, OPEN_AGE)
            years = list(range(2003, 2025))
            if pop == "TWN":                                   # add the rebuilt 2025
                x = r25[r25.sex == sex].sort_values("age")
                D = np.column_stack([D, x["deaths_moi"].to_numpy()])
                E = np.column_stack([E, x["exposure"].to_numpy()])
                years.append(2025)
            col = {y: i for i, y in enumerate(years)}
            m_obs = D / E
            targets = np.array([y for y in years if y >= 2020])
            sl = slice(col[FIT[0]], col[FIT[1]] + 1)
            fc = tl.forecast(D[:, sl], E[:, sl], targets - FIT[1], jump="corrected", n_boot=N_BOOT, n_paths=N_PATHS, rng=rng)
            central = fc["central"]                                                     # (A, H)
            sims = tl.predictive_rates(fc["draws"], E[:, [col[y] for y in targets]], rng)   # (S, H, A)
            w19 = E[:, col[2019]]
            for grouping, groups in (("band", BANDS), ("five_year", FIVE)):
                for label, (lo, hi) in groups.items():
                    idx = (AGES >= lo) & (AGES <= hi)
                    w = w19[idx] / w19[idx].sum()
                    base = float((m_obs[idx, col[2019]] * w).sum())
                    for y in range(2012, 2020):                                         # model-free series before the pandemic
                        o = float((m_obs[idx, col[y]] * w).sum())
                        rows.append((pop, sex, y, grouping, label, o * 1e5, np.nan, np.nan, np.nan, np.nan, np.nan, o / base, np.nan, np.nan, np.nan))
                    for h, y in enumerate(targets):
                        o = float((m_obs[idx, col[y]] * w).sum())
                        e = float((central[idx, h] * w).sum())
                        s = (sims[:, h, idx] * w).sum(axis=1)
                        lo_s, hi_s = np.quantile(s, [0.025, 0.975])
                        se_lo, se_hi = tl.quantile_mcse(s, [0.025, 0.975], cluster=N_PATHS, seed=MCSE_SEED)
                        tail = min(np.mean(s <= o), np.mean(s >= o))
                        p = min(1.0, max(2.0 * tail, 1.0 / len(s)))
                        # the ratio bounds are o / quantile, so their Monte Carlo error follows by the delta method
                        rows.append((pop, sex, int(y), grouping, label, o * 1e5, e * 1e5, o / e, o / hi_s, o / lo_s, p, o / base,
                                     o / hi_s * float(se_hi) / hi_s, o / lo_s * float(se_lo) / lo_s, tl.two_sided_p_mcse(s, o, N_PATHS)))
    out = pd.DataFrame(rows, columns=["population", "sex", "year", "grouping", "age_group", "observed_rate_per_100k", "expected_rate_per_100k",
                                      "ratio_to_trend", "ratio_low", "ratio_high", "p_value", "ratio_to_2019",
                                      "ratio_low_mcse", "ratio_high_mcse", "p_value_mcse"])
    out.to_csv(OUT / "age_rate_ratios.csv", index=False)
    (OUT / "age_rate_ratios_metadata.json").write_text(json.dumps({
        "script": "research/trend_age_rates.py", "fit_window": list(FIT), "jump": "corrected", "seed": SEED,
        "poisson_bootstrap_fits": N_BOOT, "paths_per_fit": N_PATHS,
        "standard": "2019 exposure of the same population and sex, within each age group",
        "interval": "all refits x paths (N_BOOT x N_PATHS simulated rates); ratio bounds are the observed rate divided by the 97.5th and 2.5th percentiles",
        "p_value": "two-sided: twice the smaller tail share of the simulated rates, not below 1 / (N_BOOT x N_PATHS)",
        "monte_carlo_error": "columns *_mcse: standard error due to the finite number of simulations.  Quantiles of the simulated rates: "
                             "100 resamples of whole refits (separate random stream); ratio bounds: delta method applied to those quantiles.  "
                             "p-values: analytic clustered approximation, not resampled (trendlib.two_sided_p_mcse)", "mcse_seed": MCSE_SEED,
        "taiwan_2025": "rebuilt rates from results_trend/taiwan_2025_rates.csv",
        "code_sha256": tl.code_sha256("research/trend_age_rates.py", "research/trendlib.py")}, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    b = out[(out.grouping == "band") & (out.year >= 2020)]
    for pop in FOCUS:
        print(f"\n== {pop}: observed / expected death rate by age band [95% interval] ==")
        for sex in tl.SEXES:
            for label in BANDS:
                g = b[(b.population == pop) & (b.sex == sex) & (b.age_group == label)]
                print(f"{sex:6s} {label:6s}", " | ".join(f"{int(r.year)}: {r.ratio_to_trend:.3f} [{r.ratio_low:.3f},{r.ratio_high:.3f}]" for r in g.itertuples()))
    print("\n== Taiwan: rate relative to 2019 by age band ==")
    t = out[(out.population == "TWN") & (out.grouping == "band") & (out.year >= 2019)]
    print(t.pivot_table(index=["sex", "age_group"], columns="year", values="ratio_to_2019").round(3).to_string())


if __name__ == "__main__":
    main()
